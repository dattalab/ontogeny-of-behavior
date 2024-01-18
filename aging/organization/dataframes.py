import re
import h5py
import numba
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from datetime import datetime
from aging.behavior.scalars import compute_scalars
from toolz import concat, keyfilter, valmap


_parser = re.compile(r"session_(\d+)")


def parse_date(path):
    return datetime.strptime(
        _parser.search(path.parents[1].name).group(1), "%Y%m%d%H%M%S"
    )


def jax_parse_date(path: Path) -> datetime:
    return datetime.strptime("_".join(path.stem.split("_")[:2]), "%Y-%m-%d_%H-%M-%S")


def create_uuid_map(folders, syllable_path, experiment) -> dict:
    uuid_map = {}
    for file in tqdm(
        filter(
            lambda f: get_experiment(f) == experiment,
            concat(f.glob("**/*results_00.h5") for f in folders),
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


def ontogeny_age_map_fun(age: str, is_female: bool = False) -> int:
    """Parses a string with age written in either week or month form, and
    converts it to week form."""
    try:
        return int(age.split("w")[0])
    except ValueError:
        female_flag_age = 72 if is_female else 78
        return {"3": 12, "6": 24, "9": 36, "12": 52, "18": female_flag_age, "22": 90}[
            age.split("m")[0]
        ]


def longtogeny_age_map_fun(age: datetime, v2: bool = False) -> float:
    if v2:  # for both males and females
        start_date = datetime(year=2023, month=6, day=29)
        initial_age = 20  # days
    else:
        start_date = datetime(year=2021, month=3, day=30)
        initial_age = 21  # days

    # subtract start date from age, convert to weeks
    delta = age - start_date
    age = (delta.days + initial_age) / 7

    return age


def jax_longtogeny_age_map_fun(age: datetime) -> float:
    start_date = datetime(year=2021, month=5, day=19)
    initial_age = 122  # 4 months in days

    # subtract start date from age, convert to weeks
    delta = age - start_date
    age = (delta.days + initial_age) / 7

    return age


def get_age(path: Path) -> int | float:
    experiment = get_experiment(path)

    if "ontogeny" in experiment:
        age = ontogeny_age_map_fun(path.parents[2].name.split("_")[0], 'female' in experiment)
    elif "longtogeny_v2" in experiment:
        age = longtogeny_age_map_fun(parse_date(path), v2=True)
    elif "longtogeny_males" == experiment:
        age = longtogeny_age_map_fun(parse_date(path))
    elif "klothos" == experiment:
        # TODO: make sure that this changes if we do long-term imaging of the klothos mice
        age = 90  # weeks
    elif "jax_longtogeny" in experiment:
        age = jax_longtogeny_age_map_fun(jax_parse_date(path))
    else:
        # TODO: add age for wheel data
        age = None
    return age


def get_experiment(path: Path):
    str_path = str(path)
    if "min" in str_path and "longtogeny_07" in str_path:
        exp = f"longtogeny_v2_{path.parents[2].name.lower()}"
    elif "wheel" in str_path.lower():
        exp = "wheel"
    elif "dlight" in str_path:
        exp = "dlight"
    elif "jackson" in str_path and "win" in str_path:
        exp = "jax_longtogeny"
    elif "Klothos" in str_path:
        exp = "klothos"
    elif "Epig" in str_path:
        exp = "epigenetic_clock"
    elif "longtogeny" in str_path:
        def _get_sex(path, depth):
            if depth < 0:
                raise ValueError("bleh")
            elif path.parents[depth].name.lower() not in ("males", "females"):
                return _get_sex(path, depth - 1)
            return path.parents[depth].name.lower()
        sex = _get_sex(path, 3)
        exp = f"longtogeny_{sex}"
    elif "Dana_ontogeny" in str_path:
        exp = f"dana_ontogeny_{path.parents[3].name.lower()}"
    elif "ontogeny" in str_path.lower() and "community" not in str_path:
        exp = path.parents[3].name.lower()
        if exp == "raw_data":
            exp = path.parents[2].name.lower()
    else:
        exp = path.parents[2].name
    return exp


def insert_nans(timestamps, data, fps=30):
    print("inserting nans - old")
    df_timestamps = np.diff(np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.round(df_timestamps / np.median(df_timestamps))

    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype("float64")

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
            insert_timestamps = timestamps[idx - 1] + np.cumsum(
                np.ones(
                    ninserts,
                )
                * 1.0
                / fps
            )
            filled_data = np.insert(
                filled_data, idx, np.ones((ninserts, nfeatures)) * np.nan, axis=0
            )
            filled_timestamps = np.insert(filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


@numba.jit(nopython=True)
def insert_nans_numba(timestamps, data, fps=30):
    timestamps = list(timestamps)
    data = list(data)
    # get the difference between timestamps, accounting for np.diff's length reduction
    df_timestamps = np.diff(np.array([timestamps[0] - 1 / fps] + timestamps))
    missing = np.round(df_timestamps / np.median(df_timestamps))
    fill_idx = np.where(missing > 1)[0]

    filled_data = data.copy()
    filled_timestamps = timestamps.copy()

    for i in fill_idx[::-1]:
        n = int(missing[i] - 1)

        time_start = (timestamps[i - 1] + np.cumsum(np.full(n, 1 / fps)))[::-1]

        for j in range(n):
            filled_data.insert(i, np.nan)
            filled_timestamps.insert(i, time_start[j])

    return np.array(filled_data), np.array(filled_timestamps)


def determine_timestamp_scale(timestamps, fps=30):
    RT_MOSEQ_SCALE = 1.25e-4  # for rt-moseq setup
    OG_SCALE = 1e-3  # for original setup

    target_sample_rate = 1 / fps

    sample_rate = np.median(np.diff(timestamps))
    scale_map = {
        RT_MOSEQ_SCALE: np.abs(sample_rate * RT_MOSEQ_SCALE - target_sample_rate),
        OG_SCALE: np.abs(sample_rate * OG_SCALE - target_sample_rate),
    }
    return min(scale_map, key=scale_map.get)


def extract_scalars(path: Path, recon_key):
    try:
        with h5py.File(path, "r") as f:
            session_name = f["metadata/acquisition/SessionName"][()].decode()
            subject_name = f["metadata/acquisition/SubjectName"][()].decode()
            true_depth = f["metadata/extraction/true_depth"][()]

            keep_scalars = list(
                filter(lambda k: "mm" in k or "px" in k, f["scalars"])
            ) + [
                "angle",
                "velocity_theta",
            ]

            ts = f["timestamps"][()]
            scale = determine_timestamp_scale(ts)
            ts *= scale

            filled_ts = insert_nans_numba(ts, ts)[-1]

            scalars = dict(
                (k, f["scalars"][k][()].astype("float64")) for k in keep_scalars
            )
            filled_scalars = valmap(lambda v: insert_nans_numba(ts, v)[0], scalars)

            frames = f[recon_key][()]
            recon_scalars = compute_scalars(frames, height_thresh=15)
            recon_scalars = valmap(
                lambda v: insert_nans_numba(ts, v.astype("float64"))[0], recon_scalars
            )
        return dict(
            true_depth=true_depth,
            session_name=session_name,
            subject_name=subject_name,
            timestamps=filled_ts - filled_ts[0],
            raw_timestamps=filled_ts,
            **filled_scalars,
            **recon_scalars,
        )
    except (OSError, KeyError) as e:
        print("Error with", str(path))
        print(e)
        return None
