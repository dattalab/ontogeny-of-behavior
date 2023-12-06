from pathlib import Path

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
]
FOLDERS = tuple(Path(f) for f in FOLDERS)