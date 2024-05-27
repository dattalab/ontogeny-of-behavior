import h5py
from pathlib import Path
from ruamel.yaml import YAML
from toolz import curry, compose


def not_extracted(file: Path, debug=False):
    if file.name.endswith('filepart'):
        return False

    extracted_files = sorted(file.parent.glob("proc*/results_00.h5"))

    if extracted := len(extracted_files) > 0:
        _extracted = []
        _complete = []
        for _file in extracted_files:
            try:
                with h5py.File(_file, 'r') as h5f:
                    list(h5f)
                _extracted.append(True)
                with open(_file.with_suffix(".yaml"), "r") as conf_f:
                    yaml = YAML(typ='safe', pure=True)
                    config = yaml.load(conf_f)
                _complete.append(config["complete"])
            except Exception as e:
                if debug:
                    print(e)
                _extracted.append(False)
                _complete.append(False)
        if all(not x for x in _extracted):
            return True
        if all(not x for x in _complete):
            return True
    return not extracted


def no_depth_doubles(file):
    '''If there are depth files with two extensions (avi and dat),
    then we want to skip the avi file.'''
    return not (file.name.endswith("avi") and file.with_suffix(".dat").exists())


def multi_filter(*filters, seq):
    return compose(*(curry(filter)(f) for f in filters))(seq)


@curry
def is_size_normalized(key, file: Path | str):
    try:
        with h5py.File(file, "r") as h5f:
            return key in h5f
    except (OSError, BlockingIOError):
        return False


@curry
def not_extract_double(parent_counts: dict, path):
    if parent_counts[str(path.parents[1])] > 1:
        # return 'proc_cleaned' == path.parent.name
        return 'proc-2024-04-25' == path.parent.name
    return True