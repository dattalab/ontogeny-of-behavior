import h5py
import click
import torch
from pathlib import Path
from toolz import partial
from tqdm.auto import tqdm
from aging.size_norm.data import Session
from aging.size_norm.lightning import predict


def predict_and_save(path, model, key):
    with h5py.File(path, "r") as h5f:
        data = h5f["frames"][()]
    data = Session(data)
    output = predict(data, model, batch_size=512, desc="Predicting")

    with h5py.File(path, "r+") as h5f:
        if key in h5f:
            del h5f[key]
        h5f.create_dataset(key, data=output, dtype="float32", compression="lzf")


def hasnt_key(path, key):
    try:
        with h5py.File(path, "r") as h5f:
            return key not in h5f
    except Exception:
        print("Error loading", path)
        return False


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--key", default="size_normalized_frames", help="Key to use for the output"
)
def main(data_path, model_path, key):
    print("Processing files from", data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)

    data_path = Path(data_path)
    if data_path.is_dir():
        for path in tqdm(
            list(
                filter(partial(hasnt_key, key=key), data_path.glob("**/results_00.h5"))
            ),
            desc="Files",
        ):
            try:
                predict_and_save(path, model, key)
            except Exception:
                continue
    else:
        predict_and_save(data_path, model, key)


if __name__ == "__main__":
    main()
