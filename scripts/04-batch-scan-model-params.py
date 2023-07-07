import uuid
import toml
import click
import random
import subprocess
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from toolz import merge, valmap
from sklearn.model_selection import ParameterGrid

script = """#!/bin/env bash
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 03:30:00
#SBATCH --output={folder}/model-scan-%j.out

source $HOME/.bashrc
conda activate aging
module load gcc/9.2.0
module load cuda/11.7
python /home/wg41/code/ontogeny/scripts/03-train-size-norm.py {config_path}
"""


def process_space(space):
    if isinstance(space, list) and all(isinstance(x, (int, float)) for x in space) and len(space) <= 3:
        if isinstance(space[0], bool):
            return space
        if isinstance(space[0], int) and space[0] < 0:
            return list(map(float, np.logspace(*space[:2], int(space[2]))))
        if isinstance(space[0], int) and len(space) == 2:
            return range(space[0], space[1] + 1)
        if isinstance(space[0], (int, float)) and len(space) == 3:
            return list(map(lambda x: x.item(), np.linspace(*space[:2], int(space[2])).astype(type(space[0]))))

    return space


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("template_path", type=click.Path(exists=True))
@click.option("--seed", default=0)
@click.option("--stage", default=1, type=int)
@click.option("--dry-run", is_flag=True)
def main(config_path, template_path, seed, stage, dry_run):
    config = toml.load(config_path)
    template = toml.load(template_path)
    # try to tie all seeds together as much as possible for reproducibility
    template['augmentation']['seed'] = seed
    random.seed(seed)

    parameter_space = config['model'][f'stage{stage}']
    grid_input = valmap(process_space, parameter_space)
    grid = list(ParameterGrid(grid_input))
    random.shuffle(grid)

    for i, parameter_set in enumerate(tqdm(grid)):
        updated_params = deepcopy(template)
        for k, v in parameter_set.items():
            if k in updated_params['model']:
                updated_params['model'][k] = v
            else:
                for param_group, _ in filter(lambda x: isinstance(x[1], dict), template['model'].items()):
                    if k in updated_params['model'][param_group]:
                        updated_params['model'][param_group][k] = v
        _uuid = str(uuid.uuid4())
        save_path = Path(config['paths']['saving']) / f"stage_{stage:02d}" / _uuid
        updated_params['paths']['saving'] = str(save_path)
        if dry_run:
            print("Parameter set", i)
            print(parameter_set)
            print('---')
            print(updated_params)
            print('---')
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "config.toml", "w") as f:
                toml.dump(updated_params, f)

            new_script = script.format(folder=save_path, config_path=save_path / "config.toml")
            with open(save_path / "run.sh", "w") as f:
                f.write(new_script)
            subprocess.run(["sbatch", str(save_path / "run.sh")])


if __name__ == "__main__":
    main()