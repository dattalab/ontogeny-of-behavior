import gc
import os
import h5py
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from multiprocess import Pool
from aging.organization.paths import FOLDERS
from aging.organization.dataframes import (
    extract_scalars,
    create_uuid_map,
    jax_parse_date,
    parse_date,
    get_age,
)


def filter_session_length(df: pd.DataFrame, min_secs=600, experiment: str = None):
    uuid_length = df.groupby("uuid", sort=False)["timestamps"].max()
    remove_uuids = uuid_length[uuid_length < min_secs].index
    df = df[~df["uuid"].isin(remove_uuids)]
    if "longtogeny" in experiment and "jax" not in experiment:
        remove_uuids = uuid_length[uuid_length > 1250].index
        df = df[~df["uuid"].isin(remove_uuids)]
    gc.collect()
    return df


def longtogeny_mouse_map(string):
    """returns mouse identity for longtogeny v1 males"""
    if "long-3-0" in string:
        return "03_01"
    if "long-3-1" in string:
        return "03_02"
    if "long-4-1" in string:
        return "04_02"

    if "nt" in string.lower():
        m = "_01"
    elif "RL" in string:
        m = "_04"
    elif "R" in string:
        m = "_02"
    elif "L" in string:
        m = "_03"

    if "ong4" in string:
        return "04" + m
    if "ong3" in string:
        return "03" + m
    if "ong2" in string:
        return "02" + m
    if "ong1" in string:
        return "01" + m

    return string[:5].lower()


def corrections(df: pd.DataFrame, experiment: str):
    if "longtogeny_males" == experiment:
        # session accidentally marked as 03_03, should be 04_03
        time = pd.Timestamp("2021-04-03 17:46:45")
        df.loc[df["date"] == time, "subject_name"] = "04_03_025"

        # remove session copies that are likely females
        fsessions = [
            pd.Timestamp("2021-10-01 15:47:32"),
            pd.Timestamp("2021-10-01 16:46:10"),
            pd.Timestamp("2021-10-01 17:22:43"),
            pd.Timestamp("2021-10-01 16:17:56"),
            pd.Timestamp("2021-10-01 17:50:37"),
        ]
        df = df[~df["date"].isin(fsessions)]
        gc.collect()
    elif "wheel" in experiment:
        df = df[df["session_name"].str.contains("wheel", case=False)]
        gc.collect()
        # this is where we add age for the wheel data, because it is dependent on the session name
        age = df['session_name'].str[6:8].astype(int)
        start = pd.Timestamp(year=2023, month=6, day=1)
        df['age'] = (df['date'] - start).dt.days / 7 + age

    return df


def add_mouse_id(df: pd.DataFrame, experiment: str):
    subject = df['subject_name']
    if "longtogeny_v2" in experiment:  # true for both males and females
        mouse = subject.str[:5].str.upper()
        df['mouse'] = mouse
    elif "longtogeny_males" == experiment:
        df['mouse'] = subject.map(longtogeny_mouse_map)
    elif "wheel" in experiment:
        df['mouse'] = subject.str[:7]
    else:
        df['mouse'] = subject
    return df


def mouse_filter(df: pd.DataFrame, experiment: str):
    keep_mice = []
    if "longtogeny_males" == experiment:
        keep_mice = [
            "01_01", "01_02", "01_03", "01_04",
            "02_01", "02_02", "02_03", "02_04",
            "03_01", "03_02", "03_03", "03_04",
            "04_01", "04_02", "04_03", "04_04",
        ]
    elif "longtogeny_v2_females" == experiment:
        keep_mice = [
            "F1_01", "F1_02", "F1_03", "F1_04",
            "F2_01", "F2_02", "F2_03", "F2_04",
            "F3_01", "F3_02", "F3_03", "F3_04",
            "F4_01", "F4_02", "F4_03", "F4_04",
            "F5_01", "F5_02", "F5_03", "F5_04",
        ]
    elif "longtogeny_v2_males" == experiment:
        keep_mice = [
            'M1_01', 'M1_02', 'M1_03', 'M1_04',
            'M2_01', 'M2_02', 'M2_03', 'M2_04',
            'M3_01', 'M3_02', 'M3_03', 'M3_04',
            'M4_01', 'M4_02', 'M4_03', 'M4_04',
            'M5_01', 'M5_02', 'M5_03', 'M5_04'
        ]
    if len(keep_mice) > 0:
        df = df[df['mouse'].isin(keep_mice)]
        gc.collect()
    return df


def aggregate_into_dataframe(experiment: str, model_path: str, recon_key: str, old: bool = True, debug: bool = False):
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {n_cpus} cpus")

    model_path = Path(model_path)
    syllable_path = model_path / "all_data_pca/syllables.h5"
    pc_path = model_path / "all_data_pca/pca_scores.h5"

    def mp_extract(args):
        uuid, path = args
        return (uuid, path, extract_scalars(path, recon_key))

    uuid_map = create_uuid_map(syllable_path, experiment, old=old, debug=debug)
    if debug:
        print(uuid_map)

    df = []
    with h5py.File(syllable_path, "r") as h5f, h5py.File(pc_path, 'r') as pca_h5, Pool(n_cpus) as pool:
        loop_fun = map if debug else pool.imap_unordered
        for uuid, path, extraction_data in tqdm(loop_fun(mp_extract, uuid_map.items())):
            if extraction_data is None:
                extraction_data = dict(session_name="", subject_name="")

            date = (
                jax_parse_date(path)
                if experiment == "jax_longtogeny"
                else parse_date(path)
            )

            age = get_age(path)
            try:
                pcs = pca_h5[f"scores/{uuid}"][()]
                pca_data = {f"pc_{i:02d}": pcs[:, i] for i in range(10)}
                _df = pd.DataFrame(
                    dict(
                        experiment=experiment,
                        file=str(path),
                        uuid=uuid,
                        date=date,
                        age=age,
                        syllables=h5f[uuid][()],
                        **pca_data,
                        **extraction_data,
                    )
                )
                _df["onsets"] = _df["syllables"].diff() != 0
                float_cols = _df.select_dtypes(include=["float64", "float32"]).columns
                _df[float_cols] = _df[float_cols].astype("float32")
                _df = _df.astype(
                    dict(
                        syllables="int16",
                        file="string",
                        experiment="string",
                        session_name="string",
                        subject_name="string",
                        uuid="string",
                    )
                )
                df.append(_df)
            except Exception as e:
                print("Error on file", path)
                print(e)
                print("-" * 25)
    if len(df) == 0:
        return None
    return pd.concat(df, ignore_index=True)


def create_usage_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    usage_df = (
        df.query("onsets")
        .groupby(["age", "mouse", "subject_name", "session_name", "uuid", "date"], sort=False)[
            "syllables"
        ]
        .value_counts(normalize=False)
    )
    usage_df = usage_df.reset_index()
    usage_matrix = usage_df.pivot_table(
        values="count",
        index=["age", "mouse", "subject_name", "session_name", "uuid", "date"],
        columns="syllables",
        fill_value=0,
    )
    return usage_matrix


def create_frame_occupancy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    usage_df = (
        df.groupby(["age", "mouse", "subject_name", "session_name", "uuid", "date"], sort=False)[
            "syllables"
        ]
        .value_counts(normalize=False)
    )
    usage_df = usage_df.reset_index()
    usage_matrix = usage_df.pivot_table(
        values="count",
        index=["age", "mouse", "subject_name", "session_name", "uuid", "date"],
        columns="syllables",
        fill_value=0,
    )
    return usage_matrix


def normalize_dataframe(df: pd.DataFrame):
    sum_of_vals = df.sum(axis="columns").to_numpy()
    df = df / sum_of_vals[:, None]
    return df


def filter_high_usage(df: pd.DataFrame):
    norm_df = normalize_dataframe(df)
    filter_idx = (norm_df > 0.2).any(axis="columns")

    return df[~filter_idx]


def filter_dataframes_by_usage(df: pd.DataFrame) -> pd.DataFrame:
    usage_df = create_frame_occupancy_dataframe(df)
    usage_df = filter_high_usage(usage_df)

    usage_df = create_usage_dataframe(
        df[df['uuid'].isin(usage_df.index.get_level_values('uuid'))]
    )
    usage_df = filter_high_usage(usage_df)
    return usage_df

