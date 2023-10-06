import re
import os
import h5py
import click
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from multiprocess import Pool
from aging.organization.paths import FOLDERS
from aging.organization.dataframes import create_uuid_map, extract_scalars


@click.command()
@click.argument("data_folder", type=click.Path(exists=True))
@click.argument(
    "experiment",
    type=click.Choice(
        [
            "longtogeny_males",
            "longtogeny_v2_males",
            "longtogeny_females",
            "longtogeny_v2_females",
            "ontogeny_males",
            "ontogeny_females",
            "wheel",
            "dlight",
        ]
    ),
)
@click.option("--data-version", type=int, default=5)
@click.option("--df-version", type=int, default=0)
@click.option("--recon-key", type=str, default="win_size_norm_frames_v4")
@click.option("--rescaled-key", type=str, default="rescaled_frames")
def main(data_folder, experiment, data_version, df_version, recon_key, rescaled_key):
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser = re.compile(r"session_(\d+)")

    def parse_date(path):
        return datetime.strptime(
            parser.search(path.parents[1].name).group(1), "%Y%m%d%H%M%S"
        )

    def mp_extract(args):
        uuid, path = args
        return (uuid, path, extract_scalars(path, recon_key, rescaled_key))

    data_folder = (
        Path(data_folder).expanduser().resolve() / f"version_{data_version:02d}"
    )
    syllable_path = data_folder / "all_data_pca/syllables.h5"

    uuid_map = create_uuid_map(FOLDERS, syllable_path, experiment)

    df = []
    pool = Pool(n_cpus)
    with h5py.File(syllable_path, "r") as h5f:
        # for uuid, path, extraction_data in map(mp_extract, tqdm(uuid_map.items(), total=len(uuid_map))):
        for uuid, path, extraction_data in tqdm(
            pool.imap_unordered(mp_extract, uuid_map.items()), total=len(uuid_map)
        ):
            if extraction_data is None:
                extraction_data = dict(session_name="", subject_name="")
            age = np.nan
            if "ontogeny" in experiment:
                age = path.parents[2].name.split("_")[0]
            date = parse_date(path)
            try:
                _df = pd.DataFrame(
                    dict(
                        experiment=experiment,
                        file=str(path),
                        syllables=h5f[uuid][()],
                        date=date,
                        uuid=uuid,
                        age=age,
                        **extraction_data,
                    )
                )
                _df['onsets'] = _df['syllables'].diff() != 0
                float_cols = _df.select_dtypes(include=["float64", "float32"]).columns
                _df[float_cols] = _df[float_cols].astype("float32[pyarrow]")
                _df = _df.astype(
                    dict(
                        syllables="int16[pyarrow]",
                        file="string[pyarrow]",
                        experiment="string[pyarrow]",
                        session_name="string[pyarrow]",
                        subject_name="string[pyarrow]",
                        uuid="string[pyarrow]",
                    )
                )
                df.append(_df)
                if len(df) % 40 == 0:
                    df = [pd.concat(df, ignore_index=True)]
                    df[0].to_parquet(
                        data_folder
                        / f"{experiment}_syllable_df_v{df_version:02d}.parquet"
                    )
            except Exception as e:
                print("Error on file:", path)
                print(e)
                print("-" * 20)
                continue
    pd.concat(df, ignore_index=True).to_parquet(
        data_folder / f"{experiment}_syllable_df_v{df_version:02d}.parquet"
    )
    print("Finished creating dataframe")
    pool.close()


if __name__ == "__main__":
    main()
