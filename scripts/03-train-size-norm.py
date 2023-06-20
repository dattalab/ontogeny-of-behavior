import toml
import click
import torch
import random
import lightning.pytorch as pl
import aging.size_norm.models as models
from pathlib import Path
from toolz import keyfilter, dissoc
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from aging.size_norm.lightning import SizeNormModel
from aging.size_norm.data import TrainingPaths, AugmentationParams

def keep(d, keys):
    return keyfilter(lambda k: k in keys, d)

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    config = toml.load(config_path)
    save_folder = Path(config['paths']['saving'])
    save_folder.mkdir(exist_ok=True, parents=True)

    paths = keep(config['paths'], ["training", "wall_noise"])
    training_paths = TrainingPaths(**paths)
    config['augmentation']['rng'] = random.Random(config['augmentation']['seed'])
    augmentation = AugmentationParams(**dissoc(config['augmentation'], 'seed'))

    model_arch = models.Autoencoder if config['model']['arch'] == 'ae' else models.UNet

    model = SizeNormModel(
        training_paths,
        augmentation,
        model=model_arch,
        channels=config['model']['channels'],
        separable=config['model']['separable'],
        batch_size=64,
        adversarial_prob=config['model']['adversarial_prob'],
        lr=config['model']['lr'],
        weight_decay=config['model']['weight_decay'],
        seed=0
    )

    ckpt_cb = ModelCheckpoint(
        save_folder,
        filename=f"{model_arch.__name__}" + "-{epoch:02d}-{val_loss:.2e}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=25,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_stopping_cb],
        precision='16-mixed' if torch.cuda.is_available() else "bf16-mixed",
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()