import h5py
import numpy as np
from pathlib import Path
from aging.organization.paths import get_experiment_grouped_files
from aging.organization.util import is_size_normalized, not_extract_double
from toolz import valmap, partial, compose, frequencies, concat


def organize_files(size_key: str, pca_path: Path | str, return_duplicates=False):

    if isinstance(pca_path, str):
        pca_path = Path(pca_path)

    size_filter = is_size_normalized(size_key)

    # gather files and remove non-size normalized files
    all_files = get_experiment_grouped_files()
    filtered_files = valmap(compose(list, partial(filter, size_filter)), all_files)

    # filter out duplicate sessions where I've re-extracted multiple times
    parent_counts = frequencies(str(f.parents[1]) for f in concat(filtered_files.values()))
    doublet_filter = not_extract_double(parent_counts)
    filtered_files = valmap(compose(list, partial(filter, doublet_filter)), filtered_files)

    duplicates = []
    for experiment, files in filtered_files.items():
        out_folder = pca_path / experiment
        out_folder.mkdir(exist_ok=True, parents=True)

        for file in files:
            new_path = out_folder / file.parents[1].with_suffix(".h5").name
            yaml_path = new_path.with_suffix(".yaml")
            try:
                new_path.symlink_to(file)
                yaml_path.symlink_to(file.with_suffix(".yaml"))
            except FileExistsError:
                duplicates.append(file)
    if return_duplicates:
        return duplicates


def apply_whitening(data, L, mu):
    return np.linalg.solve(L, (data - mu).T).T


def get_whitening_params(data_dict):
    non_nan = lambda x: x[~np.isnan(np.reshape(x, (x.shape[0], -1))).any(1)]
    meancov = lambda x: (x.mean(0), np.cov(x, rowvar=False, bias=1))

    mu, Sigma = meancov(np.concatenate(list(map(non_nan, data_dict.values()))))
    L = np.linalg.cholesky(Sigma)

    return mu, L


def get_whitening_params_from_training_data(path: Path):
    pca_path = path / "_pca/pca_scores.h5"

    with h5py.File(pca_path, 'r') as h5f:
        pc_data = {k: h5f['scores'][k][:, :10] for k in h5f['scores']}
    
    mu, L = get_whitening_params(pc_data)

    return mu, L