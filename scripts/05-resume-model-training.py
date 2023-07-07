import toml
import click
import subprocess
from pathlib import Path
from tqdm.auto import tqdm

script = """#!/bin/env bash
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH --output={folder}/resume-model-train-%j.out

source $HOME/.bashrc
conda activate aging
module load gcc/9.2.0
module load cuda/11.7
python /home/wg41/code/ontogeny/scripts/03-train-size-norm.py {config_path} --checkpoint {ckpt_path}
"""


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--max-epochs", type=int, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--test", is_flag=True)
def main(path, max_epochs, dry_run, test):
    path = Path(path)
    files = sorted(path.glob("**/*.ckpt"))
    for file in tqdm(files):
        if max_epochs is not None:
            config = toml.load(file.with_name("config.toml"))
            config['trainer']['max_epochs'] = max_epochs
            with open(file.with_name("resume_config.toml"), 'w') as f:
                toml.dump(config, f)
        new_script = script.format(folder=file.parent, config_path=str(file.with_name("resume_config.toml")), ckpt_path=str(file))
        if dry_run:
            print("Dry run on file...", file)
        else:
            with open(file.with_name("rerun.sh"), "w") as f:
                f.write(new_script)
            subprocess.run(["sbatch", str(file.with_name("rerun.sh"))])
            if test:
                print("Submitting test job...", file)
                return


if __name__ == "__main__":
    main()
