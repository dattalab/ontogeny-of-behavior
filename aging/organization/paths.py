from pathlib import Path
from dataclasses import dataclass
from toolz import groupby, concat
from aging.organization.dataframes import get_experiment

FOLDERS = [
    '/n/groups/datta/Dana/Ontogeny/raw_data/Ontogeny_females',
    '/n/groups/datta/Dana/Ontogeny/raw_data/Ontogeny_males',
    '/n/groups/datta/Dana/Ontogeny/raw_data/Dana_ontogeny/Males',
    '/n/groups/datta/Dana/Ontogeny/raw_data/Dana_ontogeny/Females',
    # '/n/groups/datta/Dana/Ontogeny/raw_data/longtogeny_pre_unet/Females',
    '/n/groups/datta/Dana/Ontogeny/raw_data/longtogeny_pre_unet/Males',
    '/n/groups/datta/min/longtogeny_072023/Males',
    '/n/groups/datta/min/longtogeny_072023/Females',
    # '/n/groups/datta/min/dominance_v1',
    # '/n/groups/datta/min/community_v1',
    '/n/groups/datta/min/wheel_062023',
    # '/n/groups/datta/min/cas_behavior_01',
    # '/n/groups/datta/min/sham_behavior_01',
    '/n/groups/datta/win/longtogeny/dlight',
    '/n/groups/datta/min/longtogeny_052023/Males',  # second round of original longtogeny expt
    '/n/groups/datta/win/longtogeny/data/jackson-labs/datta_i',  # data from jax
    # '/n/groups/datta/min/longtogeny_052023/Females',  # second round of original longtogeny expt
    '/n/groups/datta/Dana/Ontogeny/raw_data/Klothos',
    '/n/groups/datta/Dana/Ontogeny/raw_data/Epigclock',
]
FOLDERS = tuple(Path(f) for f in FOLDERS)


def get_experiment_grouped_files():
    return groupby(get_experiment, concat(f.glob("**/*results_00.h5") for f in FOLDERS))



@dataclass
class ValidationPaths:
    age_classifier: Path = Path('/n/groups/datta/win/longtogeny/size_norm/validation_data/poses_for_age_classifier.p')
    classifier_pipeline: Path = Path('/n/groups/datta/win/longtogeny/pipeline_results/age_classifier')


@dataclass
class TrainingPaths:
    tps_training_data: Path = Path('/n/groups/datta/win/longtogeny/size_norm/training_data/poses_for_tps_mapping.p.gz')
    tps_fits: Path = Path('/n/groups/datta/win/longtogeny/size_norm/training_data/tps_fits.p.gz')
    tps_multivariate_t_params: Path = Path('/n/groups/datta/win/longtogeny/size_norm/training_data/tps_multivariate_t_params.p.gz')