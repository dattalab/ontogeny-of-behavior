import os
from typing import Union
import PIL
import torch
import random
import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F
from io import BytesIO
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
)
from aging.size_norm.models import Autoencoder
from aging.size_norm.test import test


class BehaviorValidation(Callback):
    def __init__(
        self,
        validation_path: str | Path,
        log_interval: int = 15,
        max_frames: int = 8_000,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.val_path = validation_path
        self.max_frames = max_frames

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self.log_interval == 0 and trainer.current_epoch > 0:
            results = test(
                self.val_path,
                pl_module,
                Path(trainer.log_dir) / f"test_epoch_{trainer.current_epoch:03d}",
                self.max_frames,
            )
            results = results.mean(numeric_only=True)
            pl_module.log_dict(
                dict(
                    val_r2=results["fit_score"],
                    val_mse=results["fit_mse"],
                    heldout_r2=results["heldout_score"],
                    heldout_mse=results["heldout_mse"],
                    correlation=results["corr"],
                )
            )
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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.batch_size = batch_size
        self.adversarial_prob = adversarial_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.paths = paths
        self.aug_params = aug_params
        self.lr_scheduler_params = lr_scheduler_params

        self.model = torch.jit.trace(
            model(**model_params).to(device),
            torch.zeros(batch_size, 1, image_dim, image_dim, device=device),
        )
        # self.model = model(**model_params).to(device)
        self.rng = random.Random(seed)
        self.augment = Augmenter(aug_params, paths.wall_noise)

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training or self.trainer.validating:
            x = self.augment(x)
        return x, y

    def training_step(self, batch, batch_idx):
        frames, target = batch

        if self.rng.random() < self.adversarial_prob:
            yhat = batch_adversarial(
                self.model,
                frames,
                target,
                pgd_linf,
                epsilon=0.2,
                alpha=0.05,
                num_iter=10,
                randomize=True,
            )
        else:
            yhat = self.model(frames)

        loss = F.mse_loss(yhat, target)
        self.log("train_loss", loss)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        frames, target = batch
        yhat = self.model(frames)

        loss = F.mse_loss(yhat, target)
        self.log("val_loss", loss, prog_bar=True)
        return dict(loss=loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **self.lr_scheduler_params,
                ),
                monitor="val_loss",
            ),
        )

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.seed)
        dataset = SizeNormDataset(self.paths, self.aug_params)
        self.training_data, self.val_data = torch.utils.data.random_split(
            dataset, [0.96, 0.04], generator=generator
        )

    def train_dataloader(self):
        # reduce the training datset size on each epoch
        sampler = RandomSampler(
            self.training_data,
            replacement=False,
            num_samples=len(self.training_data) // 3,
        )
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
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
