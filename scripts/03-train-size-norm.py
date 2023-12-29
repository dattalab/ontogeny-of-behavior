import toml
import click
import torch
import random
import lightning.pytorch as pl
import aging.size_norm.models as models
from pathlib import Path
from aging.size_norm.lightning import SizeNormModel, BehaviorValidation
from toolz import keyfilter, dissoc, merge, valfilter, assoc, get_in
from aging.size_norm.data import TrainingPaths, AugmentationParams
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


def keep(d, keys):
    return keyfilter(lambda k: k in keys, d)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--checkpoint", is_flag=True)
@click.option("--progress", is_flag=True)
def main(config_path, checkpoint, progress):
    config = toml.load(config_path)
    save_folder = Path(config["paths"]["saving"])
    save_folder.mkdir(exist_ok=True, parents=True)
    pl.seed_everything(config["augmentation"]["seed"])

    paths = keep(
        config["paths"], ["training", "wall_noise", "validation", "age_validation"]
    )
    training_paths = TrainingPaths(**paths)
    config["augmentation"]["rng"] = random.Random(config["augmentation"]["seed"])
    augmentation = AugmentationParams(**dissoc(config["augmentation"], "seed"))

    match get_in(["model", "lightning", "arch"], config):
        case "ae":
            model_arch = models.Autoencoder
        case "unet":
            model_arch = models.UNet
        case "efficient":
            model_arch = models.EfficientAutoencoder
        case "efficient_unet":
            model_arch = models.EfficientUNet
        case "vae":
            model_arch = models.VariationalAutoencoder
        case _:
            raise ValueError(f"Unknown model architecture {config['model']['arch']}")

    model_params = merge(
        config["model"], config["model"][config["model"]["lightning"]["arch"]]
    )
    model_params = valfilter(lambda x: not isinstance(x, dict), model_params)
    if "activation" in model_params:
        model_params = assoc(
            model_params, "activation", getattr(torch.nn, model_params["activation"])
        )

    model = SizeNormModel(
        training_paths,
        augmentation,
        model=model_arch,
        model_params=model_params,
        batch_size=64 if get_in(["model", "init_channel"], config, 32) < 512 else 32,
        **dissoc(config["model"]["lightning"], "arch"),
        seed=config["augmentation"]["seed"],
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
        patience=get_in(["callbacks", "stopping", "patience"], config, 13),
        mode="min",
        verbose=True,
    )

    dynamics_cb = BehaviorValidation(
        training_paths.validation,
        log_interval=get_in(
            ["callbacks", "dynamics_validation", "log_interval"], config, 30
        ),
        max_frames=get_in(
            ["callbacks", "dynamics_validation", "max_frames"], config, 8000
        ),
        validation_type="dynamics",
    )

    age_cb = BehaviorValidation(
        training_paths.age_validation,
        log_interval=get_in(
            ["callbacks", "age_validation", "log_interval"], config, 30
        ),
        max_frames=get_in(["callbacks", "age_validation", "max_frames"], config, 8000),
        validation_type="classification",
    )

    trainer = pl.Trainer(
        max_epochs=get_in(["trainer", "max_epochs"], config, 85),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_stopping_cb, dynamics_cb, age_cb],
        precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed",
        logger=[
            CSVLogger(save_folder, name="size_norm_scan"),
            TensorBoardLogger(save_folder, name="size_norm_scan"),
        ],
        accumulate_grad_batches=1
        if get_in(["model", "init_channel"], config, 32) < 512
        else 2,
        enable_progress_bar=progress,
    )

    checkpoint_path = None
    if checkpoint:
        ckpts = sorted(save_folder.glob("*.ckpt"))
        if len(ckpts) > 0:
            checkpoint_path = ckpts[-1]

    trainer.fit(model, ckpt_path=checkpoint_path)
    print("Done training")

    if not trainer.interrupted:
        model = SizeNormModel.load_from_checkpoint(ckpt_cb.best_model_path)
        # save jit version of model
        mdl = torch.jit.trace(
            model.model.eval(),
            torch.zeros(
                model.hparams.batch_size,
                1,
                model.hparams.image_dim,
                model.hparams.image_dim,
                device=model.device,
            ),
        )
        torch.jit.save(mdl, save_folder / "model.pt")
        print("Saved model to folder", str(save_folder))


if __name__ == "__main__":
    main()
