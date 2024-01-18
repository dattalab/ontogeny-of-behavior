import cv2
import h5py
import joblib
import numpy as np
from toolz import curry
from pathlib import Path
from aging.size_norm.data import Session, clean
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


def predict_and_save(path: str | Path, model, recon_key, frames_key="frames", rescale=False, clean_noise=False):
    with h5py.File(path, "r") as h5f:
        data = h5f[frames_key][()]

    # optionally rescale animal to match 3-month old and clean frame
    if clean_noise:
        data = np.array([clean(frame, height_thresh=8, tail_ksize=9) for frame in data])
    if rescale:
        data = rescale_data(data)

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