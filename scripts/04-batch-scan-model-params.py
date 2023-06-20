import uuid
import toml
import click
import random
import subprocess
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from toolz import merge
from sklearn.model_selection import ParameterGrid

script = """#!/bin/env bash
#SBATCH -c 2
#SBATCH -n 1
#SBATCH --mem=14G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 02:15:00
#SBATCH --output={folder}/model-scan-%j.out

source $HOME/.bashrc
conda activate aging
python /home/wg41/code/ontogeny/scripts/03-train-size-norm.py {config_path}
"""


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("template_path", type=click.Path(exists=True))
@click.option("--seed", default=0)
def main(config_path, template_path, seed):
    config = toml.load(config_path)
    template = toml.load(template_path)
    # try to tie all seeds together as much as possible for reproducibility
    template['augmentation']['seed'] = seed
    random.seed(seed)

    def sample_channels(channels, channel_depth):
        depth = random.randint(*channel_depth)
        # assume that we always want to increase channel count 
        channels = [1] + sorted(random.choices(channels, k=depth))
        return tuple(channels)
    sampled_channels = list(set([sample_channels(config['model']['channels'], config['model']['channel_depth']) for _ in range(125)]))

    grid_input = dict(
        channels=sampled_channels,
        # separable=config['model']['separable'],
        # lr=list(map(float, logspace(*config['model']['lr'], 3))),
        # weight_decay=list(map(float, logspace(*config['model']['weight_decay'], 4))),
        arch=config['model']['arch'],
    )
    grid = list(ParameterGrid(grid_input))
    random.shuffle(grid)

    for parameter_set in tqdm(grid):
        updated_params = deepcopy(template)
        updated_params['model'] = merge(updated_params['model'], parameter_set)
        _uuid = str(uuid.uuid4())
        save_path = Path(config['paths']['saving']) / _uuid
        updated_params['paths']['saving'] = str(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.toml", "w") as f:
            toml.dump(updated_params, f)

        new_script = script.format(folder=save_path, config_path=save_path / "config.toml")
        with open(save_path / "run.sh", "w") as f:
            f.write(new_script)
        subprocess.run(["sbatch", str(save_path / "run.sh")])


if __name__ == "__main__":
    main()