import toml
import click
import torch
import random
import lightning.pytorch as pl
import aging.size_norm.models as models
from pathlib import Path
from aging.size_norm.lightning import SizeNormModel, BehaviorValidation
from aging.size_norm.test import test
from toolz import keyfilter, dissoc, merge, valfilter, assoc
from aging.size_norm.data import TrainingPaths, AugmentationParams
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


def keep(d, keys):
    return keyfilter(lambda k: k in keys, d)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option('--checkpoint', default=None, type=click.Path(exists=True))
def main(config_path, checkpoint):
    config = toml.load(config_path)
    save_folder = Path(config['paths']['saving'])
    save_folder.mkdir(exist_ok=True, parents=True)
    pl.seed_everything(config['augmentation']['seed'])

    paths = keep(config['paths'], ["training", "wall_noise", "validation"])
    training_paths = TrainingPaths(**paths)
    config['augmentation']['rng'] = random.Random(config['augmentation']['seed'])
    augmentation = AugmentationParams(**dissoc(config['augmentation'], 'seed'))

    match config['model']['lightning']['arch']:
        case 'ae':
            model_arch = models.Autoencoder
        case 'unet':
            model_arch = models.UNet
        case 'efficient':
            model_arch = models.EfficientAutoencoder
        case 'efficient_unet':
            model_arch = models.EfficientUNet
        case _:
            raise ValueError(f"Unknown model architecture {config['model']['arch']}")

    model_params = merge(config['model'], config['model'][config['model']['lightning']['arch']])
    model_params = valfilter(lambda x: not isinstance(x, dict), model_params)
    if 'activation' in model_params:
        model_params = assoc(model_params, 'activation', getattr(torch.nn, model_params['activation']))

    model = SizeNormModel(
        training_paths,
        augmentation,
        model=model_arch,
        model_params=model_params,
        batch_size=64,
        **dissoc(config['model']['lightning'], 'arch'),
        seed=config['augmentation']['seed']
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
        patience=config.get('callbacks', {}).get('stopping', {}).get('patience', 13),
        mode="min",
    )

    beh_val_cb = BehaviorValidation(
        training_paths.validation,
        log_interval=config.get('callbacks', {}).get('validation', {}).get('log_interval', 30),
        max_frames=config.get('callabacks', {}).get('validation', {}).get('max_frames', 8000),
    )

    trainer = pl.Trainer(
        max_epochs=config.get('trainer', {}).get('max_epochs', 85),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_stopping_cb, beh_val_cb],
        precision='16-mixed' if torch.cuda.is_available() else "bf16-mixed",
        logger=[CSVLogger(save_folder, name="size_norm_scan"), TensorBoardLogger(save_folder, name="size_norm_scan")],
    )
    trainer.fit(model, ckpt_path=checkpoint)

    model = SizeNormModel.load_from_checkpoint(ckpt_cb.best_model_path)

    # save jit version of model
    torch.jit.save(model.model, save_folder / "model.pt")

    # test the model
    # test(training_paths.validation, model, save_folder / "test")


if __name__ == '__main__':
    main()