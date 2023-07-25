import toml
import click
import subprocess
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

script = """#!/bin/env bash
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem={mem}G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --output={folder}/resume-model-train-%j.out

source $HOME/.bashrc
conda activate aging
module load gcc/9.2.0
module load cuda/11.7
python /home/wg41/code/ontogeny/scripts/03-train-size-norm.py {config_path} {ckpt}
"""


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--max-epochs", type=int, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--test", is_flag=True)
def main(path, max_epochs, dry_run, test):
    path = Path(path).expanduser().resolve()
    files = sorted(path.glob("**/*.ckpt"))
    for file in tqdm(files):
        config_path = file.with_name("config.toml")
        config = toml.load(config_path)
        if max_epochs is not None:
            config['trainer']['max_epochs'] = max_epochs
            config_path = config_path.with_name("resume_config.toml")
            with open(config_path, 'w') as f:
                toml.dump(config, f)
        # get all csv files, select last one
        csv_file = sorted(file.parent.glob("**/*.csv"))[-1]
        max_train_epoch = pd.read_csv(csv_file)['epoch'].max()
        if (max_train_epoch + 1) >= config['trainer']['max_epochs']:
            continue
        new_script = script.format(folder=file.parent, config_path=str(config_path), ckpt="--checkpoint " + str(file), mem=9)
        if dry_run:
            print(file.parent)
            print("Dry run on file...", file)
            print("Current max epoch:", max_train_epoch)
            print()
        else:
            with open(file.with_name("rerun.sh"), "w") as f:
                f.write(new_script)
            subprocess.run(["sbatch", str(file.with_name("rerun.sh"))])
            if test:
                print("Submitting test job...", file)
                return
    for file in filter(lambda f: len(list(f.parent.glob("*.ckpt"))) == 0, path.glob("**/model-scan*.out")):
        config_path = file.with_name("config.toml")
        # assume these died from OOM errors, add more memory
        new_script = script.format(folder=file.parent, config_path=str(config_path), ckpt="", mem=20)
        if not dry_run:
            with open(file.with_name("rerun.sh"), "w") as f:
                f.write(new_script)
            subprocess.run(["sbatch", str(file.with_name("rerun.sh"))])
        else:
            print("no output for folder", file.parent)


if __name__ == "__main__":
    main()
