import h5py
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from aging.behavior.scalars import compute_scalars
from toolz import concat, keyfilter, valmap, keymap


def create_uuid_map(folders, syllable_path, experiment) -> dict:
    uuid_map = {}
    for file in tqdm(
        filter(
            lambda f: get_experiment(f) == experiment,
            concat(f.glob("**/results_00.h5") for f in folders),
        )
    ):
        try:
            with h5py.File(file, "r") as h5f:
                uuid = h5f["metadata/uuid"][()].decode()
                uuid_map[uuid] = file
        except OSError:
            continue

    with h5py.File(syllable_path, "r") as h5f:
        h5f_uuids = list(h5f)
        uuid_map = keyfilter(lambda u: u in h5f_uuids, uuid_map)
    return uuid_map


def get_experiment(path: Path):
    str_path = str(path)
    if "min" in str_path and "longtogeny" in str_path:
        exp = f"longtogeny_v2_{path.parents[2].name.lower()}"
    elif "dlight" in str_path:
        return "dlight"
    elif "longtogeny" in str_path:
        sex = path.parents[3].name.lower()
        if sex not in ("males", "females"):
            sex = path.parents[2].name.lower()
            if sex not in ("males", "females"):
                raise ValueError("bleh")
        exp = f"longtogeny_{sex}"
    elif "ontogeny" in str_path.lower() and "community" not in str_path:
        exp = path.parents[3].name.lower()
        if exp == "raw_data":
            exp = path.parents[2].name.lower()
    elif "wheel" in str_path.lower():
        exp = "wheel"
    else:
        exp = path.parents[2].name
    return exp


def insert_nans(timestamps, data, fps=30):
    df_timestamps = np.diff(np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))

    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)

    if filled_data.ndim == 1:
        isvec = True
        filled_data = filled_data[:, None]
    else:
        isvec = False
    nframes, nfeatures = filled_data.shape

    for idx in fill_idx[::-1]:
        if idx < len(missing_frames):
            ninserts = int(missing_frames[idx] - 1)
            data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
            insert_timestamps = timestamps[idx - 1] + \
                np.cumsum(np.ones(ninserts,) * 1.0 / fps)
            filled_data = np.insert(filled_data, idx,
                                    np.ones((ninserts, nfeatures)) * np.nan, axis=0)
            filled_timestamps = np.insert(
                filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def extract_scalars(path: Path, recon_key, rescaled_key):
    try:
        with h5py.File(path, "r") as f:
            session_name = f["metadata/acquisition/SessionName"][()].decode()
            subject_name = f["metadata/acquisition/SubjectName"][()].decode()
            keep_scalars = list(filter(lambda k: "mm" in k or "px" in k, f["scalars"])) + [
                "angle",
                "velocity_theta",
            ]

            ts = f["timestamps"][()] / 1000
            scalars = dict((k, f["scalars"][k][()]) for k in keep_scalars)
            filled_scalars = valmap(lambda v: insert_nans(ts, v)[0], scalars)
            filled_ts = insert_nans(ts, ts)[2]

            frames = f[recon_key][()]
            # centroid = np.array(
            #     [f["scalars/centroid_x_px"][()], f["scalars/centroid_y_px"][()]]
            # ).T
            true_depth = f["metadata/extraction/true_depth"][()]
            # recon_scalars = compute_scalars(frames, centroid, true_depth)
            recon_scalars = compute_scalars(frames, height_thresh=15)
            recon_scalars = valmap(lambda v: insert_nans(ts, v)[0], recon_scalars)
            # also add rescaled scalars
            frames = f[rescaled_key][()]
            rescaled_scalars = compute_scalars(frames, is_recon=False, height_thresh=15)
            rescaled_scalars = keymap(lambda k: f"rescaled_{k}", rescaled_scalars)
            rescaled_scalars = valmap(lambda v: insert_nans(ts, v)[0], rescaled_scalars)
        return dict(
            true_depth=true_depth,
            session_name=session_name,
            subject_name=subject_name,
            timestamps=filled_ts - filled_ts[0],
            **filled_scalars,
            **recon_scalars,
            **rescaled_scalars,
        )
    except (OSError, KeyError) as e:
        print("Error with", str(path))
        print(e)
        return None