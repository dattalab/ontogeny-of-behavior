'''
NOTE: this module must be run in the `jax-moseq-og` conda environment.
'''
from jax import config
config.update("jax_enable_x64", True)
import h5py
import joblib
import jax.numpy as jnp
from pathlib import Path
from toolz import partition_all, valmap, partial
from jax_moseq.utils import batch, convert_data_precision, unbatch
from jax_moseq.models.arhmm.gibbs import resample_discrete_stateseqs
from aging.moseq_modeling.pca import apply_whitening, get_whitening_params_from_training_data


def file_access_mode(path: Path):
    if path.exists():
        try:
            h5f = h5py.File(path, 'r')
            h5f.close()
            return 'a'
        except OSError:
            return 'w'
    return 'w'


def apply_arhmm(model_folder: Path, batch_size: int = 80, robust: bool = False):
    pca_path = model_folder / "all_data_pca/pca_scores.h5"
    syllables_path = pca_path.with_name("syllables.h5")
    mode = file_access_mode(syllables_path)

    mu, L = get_whitening_params_from_training_data(model_folder)
    whitening_fun = partial(apply_whitening, L=L, mu=mu)

    model = joblib.load(model_folder / "model_params.p")
    
    with h5py.File(pca_path, 'r') as h5f, h5py.File(syllables_path, mode) as out_h5:
        uuids = h5f['scores'] if mode == 'w' else filter(lambda x: x not in out_h5.keys(), h5f['scores'])
        seq = partition_all(batch_size, uuids)

        for uuid_batch in seq:
            pc_data = {uuid: h5f['scores'][uuid][:, :10] for uuid in uuid_batch}
            pc_data = valmap(whitening_fun, pc_data)

            data = {}
            data['x'], data['mask'], (_keys, _bounds) = batch(pc_data)
            data["mask"] = jnp.where(jnp.isnan(data["x"]).any(-1), 0, data["mask"])
            data["x"] = jnp.where(jnp.isnan(data["x"]), 0, data["x"])
            data = convert_data_precision(data)
            data["mask"] = data["mask"].astype("int")

            z = resample_discrete_stateseqs(
                **data, **model, **model["params"], **model["hypparams"], robust=robust
            )

            z = unbatch(z, _keys, _bounds)
            for k, v in z.items():
                out_h5.create_dataset(k, data=v)
