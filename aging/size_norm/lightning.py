import os
from typing import Union
import PIL
import torch
import random
import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F
from io import BytesIO
from toolz import curry
from pathlib import Path
from tqdm.auto import tqdm
from ipywidgets import Image
from IPython.display import display
from lightning.pytorch import Callback
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, RandomSampler
from aging.size_norm.data import (
    AugmentationParams,
    Session,
    SizeNormDataset,
    TrainingPaths,
    Augmenter,
    unnormalize,
    AgeClassifierDataset,
    ConcatDataset,
)
from aging.size_norm.models import Autoencoder, LogisticRegression, Regression
from aging.size_norm.test import classify_age, dynamics_correlation
from torchmetrics.functional import r2_score


class BehaviorValidation(Callback):
    def __init__(
        self,
        validation_path: str | Path,
        log_interval: int = 15,
        max_frames: int = 8_000,
        validation_type: str = "classification",  # classification or dynamics
    ):
        super().__init__()
        self.log_interval = log_interval
        self.val_path = validation_path
        self.max_frames = max_frames
        self.validation_type = validation_type

    def classification(self, pl_module) -> dict:
        results = classify_age(
            AgeClassifierDataset(
                self.val_path,
                None,
                seed=pl_module.seed,
                y_type='class',
            ),
            pl_module,
        )
        return dict(age_classification=results)

    def dynamics(self, pl_module) -> dict:
        results = dynamics_correlation(self.val_path, pl_module, self.max_frames)
        return dict(dynamics_correlation=results)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # TODO: update for current method
        if trainer.current_epoch % self.log_interval == 0 and trainer.current_epoch > 0:
            if self.validation_type == "classification":
                output_dict = self.classification(pl_module)
            elif self.validation_type == "dynamics":
                output_dict = self.dynamics(pl_module)
            pl_module.log_dict(output_dict)
            pl_module.unfreeze()


class ImageDisplay(Callback):
    def __init__(self, log_interval: int = 100, num_samples: int = 5):
        super().__init__()
        self.log_interval = log_interval
        self.num_samples = num_samples
        buff = BytesIO()
        PIL.Image.fromarray(np.ones((8, 8), dtype="uint8")).save(buff, format="png")
        self.image = Image(value=buff.getvalue())
        self.displayed_image = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch,
        batch_idx: int,
    ):
        if trainer.global_step % self.log_interval == 0:
            if not self.displayed_image:
                display(self.image)
                self.displayed_image = True
            if pl_module.hparams.use_adversarial:
                (frames, target), _ = batch
            else:
                frames, target = batch
            if isinstance(frames, list):
                frames = frames[0]
            yhat = pl_module(frames.to(pl_module.device))

            frame_grid = make_grid(frames[: self.num_samples], nrow=1, normalize=True)
            target_grid = make_grid(target[: self.num_samples], nrow=1, normalize=True)
            yhat_grid = make_grid(yhat[: self.num_samples], nrow=1, normalize=True)

            grid = torch.cat((frame_grid, yhat_grid, target_grid), -1)
            _image = np.clip((grid.detach().cpu().float().numpy()) * 255, 0, 255)

            buff = BytesIO()

            PIL.Image.fromarray(_image[0].astype("uint8")).save(buff, format="png")
            self.image.value = buff.getvalue()


def pgd_linf(
    model,
    X,
    y,
    loss_fun=F.mse_loss,
    epsilon=0.1,
    alpha=0.01,
    num_iter=20,
    randomize=False,
):
    """Construct FGSM adversarial examples on the examples X
    Copied from: https://adversarial-ml-tutorial.org/adversarial_training/"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    model.eval()
    for t in range(num_iter):
        loss = loss_fun(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(
            -epsilon, epsilon
        )
        delta.grad.zero_()
    model.train()
    return delta.detach()


def batch_adversarial(model, X, y, attack, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    delta = attack(model, X, y, **kwargs)
    yp = model(X + delta)
    return yp


@curry
def exponential_scaling(current_epoch, start_value, max_value, epochs_to_max, start_epoch):
    # Calculate the scaling factor based on the time elapsed
    scaling_factor = np.log(max_value / start_value) / epochs_to_max
    
    # Calculate the scaled value at the current time
    scaled_value = start_value * np.exp(scaling_factor * (current_epoch - start_epoch))

    # Make sure the scaled value does not exceed the maximum value
    scaled_value = np.clip(scaled_value, 0, max_value)
    scaled_value *= (current_epoch >= start_epoch)

    return scaled_value


def r2_loss(input, target):
    return 1 - r2_score(input, target)


def abs_r2_loss(input, target):
    return 1 - torch.abs(r2_score(input, target))


# ----- lightning module ----- #
class SizeNormModel(pl.LightningModule):
    def __init__(
        self,
        paths: TrainingPaths,
        aug_params: AugmentationParams = AugmentationParams(),
        image_dim=80,
        model=Autoencoder,
        model_params=dict(),
        batch_size=64,
        adversarial_prob=0.3,
        lr=1e-3,
        weight_decay=1e-5,
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr_scheduler_params=dict(patience=16),
        use_adversarial: bool = False,
        adversarial_type: str = "class",  # or continuous
        adversarial_arch: str = "linear",  # or nonlinear
        adversarial_age_scaling=dict(),
        adversarial_age_lr: float = 1e-3,
        jit=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = not use_adversarial
        self.seed = seed
        self.rng = random.Random(seed)
        self.paths = paths
        self.lr_scheduler_params = lr_scheduler_params
        self.scaling_fun = exponential_scaling(**adversarial_age_scaling)

        if jit:
            self.model = torch.jit.trace(
                model(**model_params).to(device),
                torch.zeros(batch_size, 1, image_dim, image_dim, device=device),
            )
        else:
            self.model = model(**model_params).to(device)
        self.augment = Augmenter(aug_params, paths.wall_noise)

        if adversarial_type == "class" and use_adversarial:
            self.classifier = LogisticRegression(9, adversarial_arch == "linear")
            self.class_loss = F.cross_entropy
            self.combined_loss = F.cross_entropy
        elif adversarial_type == "continuous" and use_adversarial:
            self.classifier = Regression(image_dim * image_dim, adversarial_arch == "linear")
            self.class_loss = r2_loss
            self.combined_loss = abs_r2_loss

    def forward(self, x, target=None):
        if (
            self.trainer.training
            and target is not None
            and self.rng.random() < self.hparams.adversarial_prob
        ):
            yhat = batch_adversarial(
                self.model,
                x,
                target,
                pgd_linf,
                epsilon=0.2,
                alpha=0.05,
                num_iter=10,
                randomize=True,
            )
        else:
            yhat = self.model(x)
        return yhat

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training or self.trainer.validating:
            if self.hparams.use_adversarial and self.trainer.training:
                (frames, target) = x
                frames = self.augment(frames)
                x = (frames, target)
            else:
                x = self.augment(x)
        return x, y

    def train_with_classifier(self, batch):
        (frames, target), (age_frames, age_class) = batch
        sn_opt, class_opt = self.optimizers()

        # train the autoencoder with reconstruction loss and age classification loss
        self.toggle_optimizer(sn_opt)
        yhat = self(frames, target)
        loss = F.mse_loss(yhat, target)
        self.log("train_loss", loss)

        pred_class = self.classifier(self.model(age_frames)).squeeze()
        scale_factor = self.scaling_fun(self.current_epoch)
        self.log("scale_factor", scale_factor, prog_bar=True)
        class_loss = self.combined_loss(pred_class, age_class) * scale_factor

        self.manual_backward(loss - class_loss)
        sn_opt.step()
        sn_opt.zero_grad()
        self.untoggle_optimizer(sn_opt)

        # train the classifier
        self.toggle_optimizer(class_opt)
        class_opt.zero_grad()
        pred_class = self.classifier(self.model(age_frames).detach()).squeeze()
        class_loss = self.class_loss(pred_class, age_class)
        self.log("class_loss", class_loss, prog_bar=True)
        if self.hparams.adversarial_type == "class":
            self.log(
                "class_acc",
                torch.mean((age_class == pred_class.argmax(dim=1)).to(torch.float32)),
                prog_bar=True,
            )
        self.manual_backward(class_loss)
        class_opt.step()
        class_opt.zero_grad()
        self.untoggle_optimizer(class_opt)

        return dict(loss=loss, class_loss=class_loss, combined_loss=loss + class_loss)

    def training_step(self, batch, batch_idx):
        if self.hparams.use_adversarial:
            return self.train_with_classifier(batch)

        frames, target = batch
        yhat = self(frames, target)

        loss = F.mse_loss(yhat, target)
        self.log("train_loss", loss)

        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        frames, target = batch
        yhat = self(frames)

        loss = F.mse_loss(yhat, target)
        self.log("val_loss", loss, prog_bar=True)
        return dict(loss=loss)

    def on_validation_epoch_end(self):
        if self.hparams.use_adversarial and "val_loss" in self.trainer.callback_metrics:
            self.lr_schedulers().step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        opt_dict = dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **self.lr_scheduler_params,
                ),
                monitor="val_loss",
            ),
        )
        if self.hparams.use_adversarial:
            optimizer2 = torch.optim.AdamW(
                self.classifier.parameters(),
                lr=self.hparams.adversarial_age_lr,
                weight_decay=self.hparams.weight_decay,
            )
            return [opt_dict, dict(optimizer=optimizer2)]
        return opt_dict

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.seed)
        dataset = SizeNormDataset(self.paths)
        self.training_data, self.val_data = torch.utils.data.random_split(
            dataset, [0.96, 0.04], generator=generator
        )
        # set up optional age classifier
        if self.hparams.use_adversarial:
            self.classifier_data = AgeClassifierDataset(
                self.paths.age_validation,
                len(self.training_data),
                seed=self.seed,
                y_type=self.hparams.adversarial_type,
            )

    def train_dataloader(self):
        # reduce the training datset size on each epoch
        sampler = RandomSampler(
            self.training_data,
            replacement=False,
            num_samples=len(self.training_data) // 3,
        )
        return DataLoader(
            ConcatDataset(self.training_data, self.classifier_data) if self.hparams.use_adversarial else self.training_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
        )


def predict(
    data: Session,
    model: Union[str, Path, torch.nn.Module],
    batch_size=512,
    **tqdm_kwargs,
):
    dataset = DataLoader(data, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(model, torch.nn.Module):
        model = SizeNormModel.load_from_checkpoint(model, map_location=device)
        model.freeze()
    model.eval()

    output = []
    with torch.no_grad():
        for batch in tqdm(dataset, **tqdm_kwargs):
            output.append(unnormalize(model(batch.to(device))).cpu().numpy().squeeze())
    output = np.concatenate(output, axis=0)
    del dataset
    del data

    return output
