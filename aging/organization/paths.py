from pathlib import Path

FOLDERS = [
    '/n/groups/datta/Dana/Ontogeny/raw_data/Ontogeny_females',
    '/n/groups/datta/Dana/Ontogeny/raw_data/Ontogeny_males',
    # '/n/groups/datta/Dana/Ontogeny/raw_data/longtogeny_pre_unet/Females',
    '/n/groups/datta/Dana/Ontogeny/raw_data/longtogeny_pre_unet/Males',
    '/n/groups/datta/min/longtogeny_072023',
    # '/n/groups/datta/min/dominance_v1',
    # '/n/groups/datta/min/community_v1',
    '/n/groups/datta/min/wheel_062023',
    # '/n/groups/datta/min/cas_behavior_01',
    # '/n/groups/datta/min/sham_behavior_01',
    '/n/groups/datta/win/longtogeny/dlight'
]
FOLDERS = tuple(Path(f) for f in FOLDERS)