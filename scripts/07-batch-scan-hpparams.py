import uuid
import toml
import click
import shutil
import numpy as np
from pathlib import Path
from toolz import dissoc, first, merge, valmap
from sklearn.model_selection import ParameterGrid


def parse_parameter_name(name: str, template: dict):
    parts = name.split("_")
    assert any(
        x.startswith(parts[0]) for x in template
    ), f"parameter name {parts[0]} not found in template"
    if "_z" in name:
        # do something else
        pass
    # if this is supposed to apply to an x/y variable
    elif not any(f"_{x}" in name for x in ("x", "y")):
        # apply range to each augmentation parameter
        is_parameter = lambda s: any(s == f"{parts[0]}_{x}" for x in ("x", "y"))
        apply_keys = [x for x in template if is_parameter(x)]

    bound = None
    if parts[-1] in ("low", "high"):
        bound = parts[-1]

    return apply_keys, bound


def set_ranged_value(value, bound, template_value):
    """Set the appropriate value from within the template guided by the bound"""
    if isinstance(value, np.float_):
        value = float(value)
    elif isinstance(value, np.int_):
        value = int(value)
    if bound == "low":
        return [value, template_value[1]]
    elif bound == "high":
        return [template_value[0], value]
    else:
        return value


def apply_grid_param(name: str, val, template: dict):
    apply_keys, bound = parse_parameter_name(name, template)
    return {k: set_ranged_value(val, bound, template[k]) for k in apply_keys}


def transform_parameter(values):
    if isinstance(values, (list, tuple)):
        return values
    elif isinstance(values, dict):
        val_fun = lambda d: dissoc(d, "type")
        if "num" in values:
            arr_fun = (
                np.linspace
                if values.get("type", "linear") == "linear"
                else np.logspace
            )
        elif "step" in values:
            arr_fun = np.arange
        return arr_fun(**val_fun(values))
    elif isinstance(values, (int, float)):
        return [values]


def generate_parameter_space(config: dict):
    # top level is parameter container (i.e., augmentation, model, etc.)
    space = {}
    for container, params in config.items():
        agg = valmap(transform_parameter, params)
        space[container] = agg
    return space


def update_augmentation_dict(item, modification_dict, debug=False):
    new_param_set = []
    for k, v in item.items():
        new_param_set.append(apply_grid_param(k, v, modification_dict))
    new_param_dict = merge(modification_dict, *new_param_set)
    if debug:
        print("Param set", item)
        print("New param set", new_param_set)
        print("New param dict", new_param_dict)
    return new_param_dict


def _recursive_update(d: dict, key, new_value):
    if key in d:
        d[key] = new_value
    else:
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = _recursive_update(v, key, new_value)
    return d


def constrained_merge(modification_dict: dict, item: dict) -> dict:
    '''Recursively merge item into modification_dict only if the key in item
    is present in modification_dict at some level of the dictionary.'''
    agg_dict = modification_dict.copy()
    for k, v in item.items():
        assert not isinstance(v, dict), "item cannot contain nested dictionaries"
        agg_dict = _recursive_update(agg_dict, k, v)
    return agg_dict


def update_model_dict(item, modification_dict, debug=False):
    new_param_dict = constrained_merge(modification_dict, item)
    if debug:
        print("Param set", item)
        print("New param dict", new_param_dict)
    return new_param_dict


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
    assert (
        "paths" in config
    ), "parameter space must contain a paths section to save the configs"
    save_path = Path(config["paths"]["saving"]) / f"stage_{stage:02d}"

    # if save_path exists and contains files, delete contents to reset
    if save_path.exists() and reset_run:
        shutil.rmtree(save_path)

    template["augmentation"]["seed"] = seed

    # for this version, should only contain augmentation parameters
    parameter_config = config[f"stage{stage}"]
    assert (
        len(parameter_config) == 1
    ), "parameter space can only contain one group of parameters (i.e., model OR augmentation)"
    parameter_space = generate_parameter_space(parameter_config)
    if debug:
        print("Param space", parameter_space)
    parameter_group = first(parameter_space)

    # generate parameter grid
    grid = list(ParameterGrid(first(parameter_space.values())))
    if debug:
        print("Grid", grid)
    
    modification_dict = template[parameter_group].copy()
    # augmentation_dict = template["augmentation"]
    new_config_files = []
    for item in grid:
        if parameter_group == "augmentation":
            new_param_dict = update_augmentation_dict(item, modification_dict, debug)
        elif parameter_group == "model":
            new_param_dict = update_model_dict(item, modification_dict, debug)
        if dry_run:
            print(new_param_dict)
        else:
            uuid_path = save_path / str(uuid.uuid4())
            uuid_path.mkdir(parents=True, exist_ok=True)
            template['paths']['saving'] = str(uuid_path)
            with open(uuid_path / "config.toml", "w") as f:
                toml.dump(
                    merge(
                        template,
                        {parameter_group: new_param_dict},
                    ),
                    f,
                )
            new_config_files.append(str(uuid_path / "config.toml"))

    # aggregate config paths to a file, print file
    if not dry_run:
        with open(save_path / "config_paths.txt", "a+") as f:
            f.write("\n".join(new_config_files))

    print(str(save_path / "config_paths.txt"))


if __name__ == "__main__":
    main()
