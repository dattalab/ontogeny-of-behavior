import toml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from toolz import get_in, valmap, groupby, compose


def get_mag(s):
    if s == 'K':
        return 1e3
    elif s == 'M':
        return 1e6
    elif s == 'B':
        return 1e9
    else:
        raise NotImplementedError()


def load_model_parameters(files: list[Path], debug=False):
    results = []
    for f in files:
        try:
            folder = f.parents[2]
            config = toml.load(folder / "config.toml")
            mse_df = pd.read_csv(f)
            tmp = mse_df.groupby('epoch').mean()
            out_files = list(folder.glob('model*.out'))
            if len(out_files) > 0:
                with open(out_files[0], 'r') as out_f:
                    for line in out_f.readlines():
                        if "Total params" in line:
                            params = line.split(' ')[:2]
                            params = float(params[0]) * get_mag(params[1])
            else:
                params = np.nan
            arch = config['model']['lightning']['arch']
            out = dict(
                depth=config['model']['depth'],
                arch=arch,
                lr=config['model']['lightning']['lr'],
                weight_decay=config['model']['lightning']['weight_decay'],
                adversarial_prob=round(config['model']['lightning']['adversarial_prob'], 2),
                channel_scaling=config['model']['channel_scaling'],
                depth_scaling=config['model']['depth_scaling'],
                double_conv=get_in(['model', arch, 'double_conv'], config, None),
                residual=get_in(['model', arch, 'residual'], config, None),
                init_depth=config['model']['init_depth'],
                init_channel=config['model']['init_channel'],
                val_mse=None if 'val_loss' not in mse_df.columns else mse_df['val_loss'].min(),
                vae_val_mse=None if 'val_mse_loss' not in mse_df.columns else mse_df['val_mse_loss'].min(),
                uuid=folder.name,
                train_mse=None if 'train_loss' not in mse_df.columns else tmp['train_loss'].min(),
                epoch=mse_df['epoch'].max(),
                param_count=params,
                file=str(folder / "model.pt"),
                activation=config['model']['activation'],
                bottleneck=get_in(['model', arch, 'bottleneck'], config, None),
                curriculum_learning=get_in(['model', 'lightning', 'use_curriculum_learning'], config, False),
                use_fft=get_in(['model', arch, 'use_fft_branch'], config, False),
            )
            if 'dynamics_correlation' in mse_df.columns:
                out['dynamics_corr'] = mse_df['dynamics_correlation'].dropna().iloc[-1]
            if 'age_classification' in mse_df.columns:
                out['age_class'] = mse_df['age_classification'].dropna().iloc[-1]
            results.append(out)
        except Exception as e:
            if debug:
                print(e)
    results = pd.DataFrame(results)
    return results


def flatten(x):
    return x.reshape(len(x), -1)


def subsample(data, subset=200, seed=0):
    rng = np.random.default_rng(seed)
    sample = []
    for v in data.values():
        sample.append(v[rng.permutation(len(v))[:subset]])
    return np.concatenate(sample, axis=0)


def multi_stage_pca(data, subset_frames=200, seed=0):
    '''Data is a dict where each key is a tuple of (age, session_path)'''

    # group by age, keep just values
    grouped_data = groupby(lambda x: x[0][0], data.items())
    grouped_data = valmap(lambda l: np.concatenate(list(map(lambda x: x[1], l)), axis=0), grouped_data)

    train = subsample(grouped_data, subset=subset_frames, seed=seed)
    pca = PCA(n_components=10).fit(flatten(train))

    apply_pca = compose(pca.transform, flatten)

    pcs = valmap(
        apply_pca,
        data,
    )

    return pcs
