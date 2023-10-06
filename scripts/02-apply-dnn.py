import cv2
import h5py
import click
import torch
import joblib
import numpy as np
from pathlib import Path
from toolz import curry
from tqdm.auto import tqdm
from aging.size_norm.data import Session
from aging.size_norm.lightning import predict


def compute_frame_avg(frames, thresh=9):
    peaks = np.quantile(frames.reshape(len(frames), -1), 0.95, axis=1)
    peaks -= np.median(peaks)
    med_frame = np.median(frames[peaks < thresh], axis=0)
    return med_frame


def rescale(scale, img):
    '''scale input image'''
    warp_mat = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), 0, scale)
    return cv2.warpAffine(img, warp_mat, img.shape, flags=cv2.INTER_LINEAR)


def rescale_data(frames):
    coef = joblib.load('/n/groups/datta/win/longtogeny/data/metadata/extraction_scaling.p')['coef']
    area = (compute_frame_avg(frames) > 10).sum()
    scale_factor = np.polyval(coef, area)
    frames = np.array([rescale(scale_factor, im) for im in frames])
    return frames


def predict_and_save(path, model, recon_key, frames_key="frames"):
    with h5py.File(path, "r") as h5f:
        # rescale animal to match 3-month old
        data = rescale_data(h5f[frames_key][()])
    data = Session(data)
    output = predict(data, model, batch_size=512, desc="Predicting")

    with h5py.File(path, "r+") as h5f:
        if recon_key in h5f:
            del h5f[recon_key]
        h5f.create_dataset(recon_key, data=output, dtype="float32", compression="lzf")


@curry
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
@click.option(
    "--frames-key",
    default="frames",
    help="Key to use for loading in depth frames as input",
)
@click.option("--force-rerun", is_flag=True)
def main(data_path, model_path, key, frames_key, force_rerun):
    print("Processing files from", data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)

    data_path = Path(data_path)
    if data_path.is_dir():
        seq = data_path.glob("**/results_00.h5")
        if not force_rerun:
            seq = filter(hasnt_key(key=key), seq)
        for path in tqdm(list(seq), desc="Files"):
            try:
                predict_and_save(path, model, recon_key=key, frames_key=frames_key)
            except Exception as e:
                print()
                print("Failure on file", str(path))
                print(e)
                continue
    else:
        predict_and_save(data_path, model, recon_key=key, frames_key=frames_key)


if __name__ == "__main__":
    main()
