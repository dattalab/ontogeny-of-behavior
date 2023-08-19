import toml
import pandas as pd
from pathlib import Path
from toolz import get_in


def get_mag(s):
    if s == 'K':
        return 1e3
    elif s == 'M':
        return 1e6
    elif s == 'B':
        return 1e9
    else:
        raise NotImplementedError()


def load_model_parameters(files: list[Path]):
    results = []
    for f in files:
        try:
            folder = f.parents[2]
            config = toml.load(folder / "config.toml")
            mse_df = pd.read_csv(f)
            tmp = mse_df.groupby('epoch').mean()
            with open(list(folder.glob('model*.out'))[0], 'r') as out_f:
                for line in out_f.readlines():
                    if "Total params" in line:
                        params = line.split(' ')[:2]
                        params = float(params[0]) * get_mag(params[1])
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
                uuid=folder.name,
                train_mse=tmp['train_loss'].min(),
                epoch=mse_df['epoch'].max(),
                param_count=params,
                file=str(folder / "model.pt"),
                activation=config['model']['activation'],
            )
            if 'dynamics_correlation' in mse_df.columns:
                out['dynamics_corr'] = mse_df['dynamics_correlation'].dropna().iloc[-1]
            if 'age_classification' in mse_df.columns:
                out['age_class'] = mse_df['age_classification'].dropna().iloc[-1]
            results.append(out)
        except Exception:
            continue
    results = pd.DataFrame(results)
    return results