import uuid
import toml
import click
import shutil
import random
import numpy as np
from pathlib import Path
from toolz import dissoc, first, merge
from sklearn.model_selection import ParameterGrid


def parse_parameter_name(name: str, template: dict):
    parts = name.split('_')
    assert any(x.startswith(parts[0]) for x in template), f"parameter name {parts[0]} not found in template"
    if '_z' in name:
        # do something else
        pass
    # if this is supposed to apply to an x/y variable
    elif not any(f"_{x}" in name for x in ('x', 'y')):
        # apply range to each augmentation parameter
        is_parameter = lambda s: any(s == f"{parts[0]}_{x}" for x in ('x', 'y'))
        apply_keys = [x for x in template if is_parameter(x)]
    
    bound = None
    if parts[-1] in ('low', 'high'):
        bound = parts[-1]

    return apply_keys, bound


def set_ranged_value(value, bound, template_value):
    '''Set the appropriate value from within the template guided by the bound'''
    if isinstance(value, np.float_):
        value = float(value)
    elif isinstance(value, np.int_):
        value = int(value)
    if bound == 'low':
        return [value, template_value[1]]
    elif bound == 'high':
        return [template_value[0], value]
    else:
        return value


def apply_grid_param(name: dict, val, template: dict):
    apply_keys, bound = parse_parameter_name(name, template)
    return {k: set_ranged_value(val, bound, template[k]) for k in apply_keys}


def generate_parameter_space(config: dict):
    # top level is parameter container (i.e., augmentation, model, etc.)
    space = {}
    for container, params in config.items():
        agg = {}
        for param, values in params.items():
            if isinstance(values, list):
                agg[param] = values
            elif isinstance(values, dict):
                val_fun = lambda d: dissoc(d, 'type')
                if 'num' in values:
                    arr_fun = np.linspace if values.get('type', 'linear') == 'linear' else np.logspace
                elif 'step' in values:
                    arr_fun = np.arange
                agg[param] = arr_fun(**val_fun(values))
        space[container] = agg
    return space



@click.command()
@click.argument("template_path", type=click.Path(exists=True))
@click.argument("parameter_space_path", type=click.Path(exists=True))
@click.option("--seed", default=0)
@click.option("--stage", default=1, type=int)
@click.option("--dry-run", is_flag=True)
@click.option("--reset-run", is_flag=True)
@click.option("--debug", is_flag=True)
def main(template_path, parameter_space_path, seed, stage, dry_run, reset_run, debug):
    template = toml.load(template_path)
    config = toml.load(parameter_space_path)
    assert "paths" in config, "parameter space must contain a paths section to save the configs"
    save_path = Path(config["paths"]["saving"]) / f"stage_{stage:02d}"

    # if save_path exists and contains files, delete contents to reset
    if save_path.exists() and reset_run:
        shutil.rmtree(save_path)

    template['augmentation']['seed'] = seed
    # TODO: load in template config and parameter space config

    # for this version, should only contain augmentation parameters
    parameter_config = config[f"stage{stage}"]
    assert len(parameter_config) == 1, "parameter space should only contain one set of container parameters"
    parameter_space = generate_parameter_space(parameter_config)
    if debug:
        print("Param space", parameter_space)

    # generate parameter grid
    grid = list(ParameterGrid(first(parameter_space.values())))
    if debug:
        print("Grid", grid)
    augmentation_dict = template['augmentation']
    new_config_files = []
    for item in grid:
        new_param_set = []
        for k, v in item.items():
            new_param_set.append(apply_grid_param(k, v, augmentation_dict))
        if debug:
            print("Param set", item)
            print("New param set", new_param_set)
        new_aug_dict = merge(augmentation_dict, *new_param_set)
        if dry_run:
            print(new_aug_dict)
        else:
            uuid_path = save_path / str(uuid.uuid4())
            uuid_path.mkdir(parents=True, exist_ok=True)
            with open(uuid_path / "config.toml", "w") as f:
                toml.dump(merge(template, {'augmentation': new_aug_dict}), f)
            new_config_files.append(str(uuid_path / "config.toml"))

    # aggregate config paths to a file, print file  
    with open(save_path / "config_paths.txt", "a+") as f:
        f.write("\n".join(new_config_files))

    print(str(save_path / "config_paths.txt"))


if __name__ == "__main__":
    main()