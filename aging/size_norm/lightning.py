import os
import PIL
from lightning import LightningModule
import torch
import random
import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F
from io import BytesIO
from typing import Any
from toolz import curry, partial, keymap
from pathlib import Path
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm
from ipywidgets import Image
from IPython.display import display
from lightning.pytorch import Callback
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, RandomSampler
from lightning.pytorch.callbacks import BaseFinetuning
from aging import organization
from aging.size_norm.data import (
    AugmentationParams,
    Session,
    SizeNormDataset,
    TrainingPaths,
    Augmenter,
    unnormalize,
    AgeClassifierDataset,
    ConcatDataset,
    CurriculumPipeline
)
from aging.size_norm.models import Autoencoder, LogisticRegression, Regression
from aging.size_norm.test import classify_age, dynamics_correlation
from aging.size_norm.vae_losses import vae_loss
from torchmetrics.functional import r2_score


class ChainedScheduler(torch.optim.lr_scheduler.ChainedScheduler):
    def step(self, metrics, epoch=None):
        for scheduler in self._schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics, epoch)
            else:
                scheduler.step()
        self._last_lr = [group['lr'] for group in self._schedulers[-1].optimizer.param_groups]

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
                y_type="class",
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
            if pl_module.hparams.train_adversarial:
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
        out = model(X + delta)
        if isinstance(out, tuple):
            out = out[0]
        loss = loss_fun(out, y)
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
def exponential_scaling(
    current_epoch, start_value, max_value, epochs_to_max, start_epoch
):
    # Calculate the scaling factor based on the time elapsed
    scaling_factor = np.log(max_value / start_value) / epochs_to_max

    # Calculate the scaled value at the current time
    scaled_value = start_value * np.exp(scaling_factor * (current_epoch - start_epoch))

    # Make sure the scaled value does not exceed the maximum value
    scaled_value = np.clip(scaled_value, 0, max_value)
    scaled_value *= current_epoch >= start_epoch

    return scaled_value


def r2_loss(input, target):
    return 1 - r2_score(input, target)


def abs_r2_loss(input, target):
    return 1 - torch.abs(r2_score(input, target))


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


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
        train_adversarial: bool = False,
        adversarial_type: str = "class",  # or continuous
        adversarial_arch: str = "linear",  # or nonlinear
        adversarial_age_scaling=dict(),
        adversarial_init_epoch: int = 0,
        adversarial_age_lr: float = 1e-3,
        max_grad: float = 100,
        use_l2_regularization_class: bool = False,
        l2_regularization_class_alpha: float = 1e-6,
        use_l2_regularization_sn: bool = False,
        l2_regularization_sn_alpha: float = 1e-6,
        vae_loss_params=dict(),
        tps_training_paths=organization.paths.TrainingPaths(),
        use_curriculum_learning: bool = False,
        curriculum_blocks: list[int] = [7000, 13000, 19000],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = not train_adversarial
        self.seed = seed
        self.rng = random.Random(seed)
        self.paths = paths
        self.lr_scheduler_params = lr_scheduler_params
        self.scaling_fun = exponential_scaling(**adversarial_age_scaling)

        self.model = model(**model_params).to(device)
        self.augment = CurriculumPipeline(1e-4, aug_params, curriculum_blocks)

        self.loss_fun = (
            F.mse_loss
            if not "Variational" in model.__name__
            else partial(vae_loss, **vae_loss_params)
        )

        if train_adversarial:
            if adversarial_type == "class":
                self.classifier = LogisticRegression(9, adversarial_arch == "linear")
                self.class_loss = self.adversarial_loss = F.cross_entropy
            elif adversarial_type == "continuous":
                self.classifier = Regression(
                    image_dim * image_dim, adversarial_arch == "linear"
                )
                self.class_loss = F.mse_loss
                self.adversarial_loss = abs_r2_loss

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
            if self.hparams.train_adversarial and self.trainer.training:
                (frames, target) = x
                if self.hparams.use_curriculum_learning:
                    frames = self.augment(frames, self.global_step)
                else:
                    frames = self.augment(frames)
                x = (frames, target)
            elif self.trainer.validating and self.hparams.use_curriculum_learning:
                x = self.augment(x, -1)
            else:
                if self.hparams.use_curriculum_learning:
                    x = self.augment(x, self.global_step)
                else:
                    x = self.augment(x, -1)
        return x, y

    def train_classifier(self, frames, y, opt):
        self.toggle_optimizer(opt)
        opt.zero_grad()
        pred_class = self.classifier(
            torch.where(frames.detach() > 0.015, frames.detach(), 0)
        ).squeeze()
        class_loss = self.class_loss(pred_class, y)
        self.log("class_loss", class_loss, prog_bar=True)
        if self.hparams.adversarial_type == "class":
            self.log(
                "class_acc",
                torch.mean((y == pred_class.argmax(dim=1)).float()),
            )
        l2_loss = torch.cat([x.view(-1) for x in self.classifier.parameters()])
        self.log("90th_percentile_weight_class", torch.quantile(l2_loss, 0.9))
        if self.hparams.use_l2_regularization_class:
            l2_loss = torch.norm(l2_loss, 2) / np.sqrt(len(l2_loss))
            self.log("class_l2_loss", l2_loss)
            class_loss += self.hparams.l2_regularization_class_alpha * l2_loss
        self.manual_backward(class_loss)
        self.clip_gradients(
            opt,
            gradient_clip_val=self.hparams.max_grad,
            gradient_clip_algorithm="norm",
        )
        grads = [
            torch.norm(p.grad, 2)
            for p in self.classifier.parameters()
            if p.grad is not None
        ]
        self.log("class_norm", sum(grads) / len(grads))
        opt.step()
        self.untoggle_optimizer(opt)

    def train_with_classifier(self, batch):
        (frames, target), (age_frames, age_class) = batch
        sn_opt, class_opt = self.optimizers()

        # train the autoencoder with reconstruction loss and age classification loss
        self.toggle_optimizer(sn_opt)
        sn_opt.zero_grad()
        yhat = self(frames, target)
        loss = F.mse_loss(yhat, target)
        self.log("train_loss", loss)

        class_loss = 0
        if (
            self.hparams.adversarial_init_epoch
            + (self.hparams.adversarial_age_scaling["start_epoch"])
            <= self.current_epoch
        ):
            age_yhat = self.model(age_frames)
            pred_class = self.classifier(
                torch.where(age_yhat > 0.015, age_yhat, 0)
            ).squeeze()
            #  scaling fun epoch start is relative to adversarial training init epoch
            scale_factor = self.scaling_fun(
                self.current_epoch - self.hparams.adversarial_init_epoch
            )
            self.log("scale_factor", scale_factor)
            class_loss = self.adversarial_loss(pred_class, age_class) * scale_factor

        l2_loss = torch.cat([x.view(-1) for x in self.model.parameters()])
        self.log("90th_percentile_weight_sn", torch.quantile(l2_loss, 0.9))
        if self.hparams.use_l2_regularization_sn:
            l2_loss = torch.norm(l2_loss, 2) / np.sqrt(len(l2_loss))
            self.log("sn_l2_loss", l2_loss)
            loss += self.hparams.l2_regularization_sn_alpha * l2_loss

        self.manual_backward(loss - class_loss)
        # clip gradients
        self.clip_gradients(
            sn_opt,
            gradient_clip_val=self.hparams.max_grad,
            gradient_clip_algorithm="norm",
        )
        grads = [
            torch.norm(p.grad, 2) for p in self.model.parameters() if p.grad is not None
        ]
        self.log("sn_norm", sum(grads) / len(grads))
        sn_opt.step()
        self.untoggle_optimizer(sn_opt)

        # train the classifier
        if self.hparams.adversarial_init_epoch <= self.current_epoch:
            if (
                self.hparams.adversarial_init_epoch
                + (self.hparams.adversarial_age_scaling["start_epoch"])
                <= self.current_epoch
            ):
                self.train_classifier(age_yhat, age_class, class_opt)
            else:
                self.train_classifier(self.model(age_frames), age_class, class_opt)

        return dict(loss=loss, class_loss=class_loss, combined_loss=loss + class_loss)

    def training_step(self, batch, batch_idx):
        if self.hparams.train_adversarial:
            return self.train_with_classifier(batch)

        frames, target = batch
        yhat = self(frames, target)

        loss = self.loss_fun(yhat, target)
        # vae loss returns a tuple
        if isinstance(loss, tuple):
            self.log_dict(loss[1])
            return dict(loss=loss[0])
        else:
            self.log("mse_loss", loss)
            self.log("train_loss", loss)
            return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        frames, target = batch
        yhat = self(frames)

        loss = self.loss_fun(yhat, target)
        if isinstance(loss, tuple):
            self.log_dict(keymap(lambda k: "val_" + k, loss[1]))
            self.log("val_loss", loss[0], prog_bar=True)
        else:
            self.log("val_mse_loss", loss)
            self.log("val_loss", loss, prog_bar=True)
        if batch_idx == 0:
            if isinstance(yhat, tuple):
                yhat = yhat[0]
            self.log_tb_images(frames, target, yhat, batch_idx)
        return dict(loss=loss)

    def log_tb_images(self, image, y_true, y_pred, batch_idx) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            return
        frame_grid = make_grid(image[:5], nrow=1, normalize=True)
        target_grid = make_grid(y_true[:5], nrow=1, normalize=True)
        yhat_grid = make_grid(y_pred[:5], nrow=1, normalize=True)
        grid = torch.cat((frame_grid, yhat_grid, target_grid), -1)
        _image = grid.detach().cpu().float().numpy()

        tb_logger.add_image(f"Image/{batch_idx}_{self.current_epoch}", _image, 0)

    def on_validation_epoch_end(self):
        if (
            self.hparams.train_adversarial
            and "val_loss" in self.trainer.callback_metrics
        ):
            self.lr_schedulers().step(self.trainer.callback_metrics["val_loss"])

    def lr_scheduler_step(self, scheduler, metric: Any | None) -> None:
        if 'val_loss' in self.trainer.callback_metrics:
            scheduler.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        params = add_weight_decay(self.model, self.hparams.weight_decay)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=0,
        )

        scheduler = ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=5),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **self.lr_scheduler_params,
            ),
        ])

        opt_dict = dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor="val_loss",
            ),
        )
        if self.hparams.train_adversarial:
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
        if self.hparams.train_adversarial:
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
            ConcatDataset(self.training_data, self.classifier_data)
            if self.hparams.train_adversarial
            else self.training_data,
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
            shuffle=True,
        )


def predict(
    data: Session,
    model: str | Path | torch.nn.Module,
    batch_size=512,
    **tqdm_kwargs,
):
    dataset = DataLoader(data, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(model, torch.nn.Module):
        if isinstance(model, Path) and model.suffix == ".pt":
            model = torch.jit.load(model, map_location=device)
        elif isinstance(model, str) and model.endswith(".pt"):
            model = torch.jit.load(model, map_location=device)
        else:
            model = SizeNormModel.load_from_checkpoint(model, map_location=device)
            model.freeze()
    model.eval()

    output = []
    with torch.no_grad():
        for batch in tqdm(dataset, **tqdm_kwargs):
            model_output = model(batch.to(device))
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            output.append(unnormalize(model_output).cpu().numpy().squeeze())
    output = np.concatenate(output, axis=0)
    del dataset
    del data

    return output


class AgingModel(pl.LightningModule):
    def __init__(
        self, paths: TrainingPaths, pred_type, sn_model, batch_size=64, lr=1e-3, seed=0
    ):
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.paths = paths
        self.sn_model = sn_model
        self.pred_type = pred_type
        self.batch_size = batch_size

        if pred_type == "class":
            self.classifier = LogisticRegression(9, linear=False)
            self.loss_fun = F.cross_entropy
        elif pred_type == "continuous":
            self.classifier = Regression(80 * 80, linear=False)
            self.loss_fun = F.mse_loss

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_fun(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_fun(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.classifier.parameters(), lr=self.lr, weight_decay=1e-5
        )

    def setup(self, stage=None):
        dataset = AgeClassifierDataset(
            self.paths.age_validation,
            None,
            seed=self.seed,
            y_type=self.pred_type,
            model=self.sn_model,
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.training_data, self.val_data = torch.utils.data.random_split(
            dataset, [0.8, 0.2], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
        )


class SizeNormModelStep2(pl.LightningModule):
    def __init__(self, paths: TrainingPaths, sn_model, age_model, lr=1e-3, seed=0):
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.sn_model = sn_model
        self.age_model = age_model
        self.paths = paths
        # turn off gradient accumulation for age model
        for p in self.age_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.sn_model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.sn_model.parameters(), lr=self.lr, weight_decay=1e-5
        )

    def setup(self, stage=None):
        dataset = SizeNormDataset(
            self.paths.size_norm_validation,
            None,
            seed=self.seed,
            model=self.age_model,
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.training_data, self.val_data = torch.utils.data.random_split(
            dataset, [0.8, 0.2], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_data,
            batch_size=64,
            shuffle=True,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=64,
            shuffle=False,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
        )


class EncoderFinetuningCallback(BaseFinetuning):
    def __init__(self, freeze_at_epoch=30):
        super().__init__()
        self._freeze_at_epoch = freeze_at_epoch
    
    def freeze_before_training(self, pl_module: LightningModule) -> None:
        return None
    
    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer) -> None:
        if epoch == self._freeze_at_epoch:
            self.freeze(modules=pl_module.model.decoder, train_bn=True)
            if "Variational" in type(pl_module.model).__name__:
                self.freeze(modules=pl_module.model.bottleneck.up, train_bn=True)