'''
Contains data filtering and cleaning functions for the longtogeny experiments.
These experiments are performed on the same cohort of mice across their lifespan.
'''
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Union


@dataclass
class LongtogenyPaths:
    strain: Union[str, Path] = Path('/n/groups/datta/rockwell/k2_data_gen4/longtogeny/LONG_AND_ONT')
    results_file: Union[str, Path] = Path('/n/groups/datta/Dana')
    pc_scores: Union[str, Path] = Path('/n/groups/datta/rockwell/k2_data_gen4/longtogeny/LONG_AND_ONT/220323_pca/pca_scores.h5')
    index_file: Union[str, Path] = Path('/n/groups/datta/rockwell/k2_data_gen4/longtogeny/LONG_AND_ONT/220323_long_and_ont_moseq6-index.yaml')
    model_file: Union[str, Path] = Path('/n/groups/datta/rockwell/k2_data_gen4/longtogeny/LONG_AND_ONT/220323_long_ont_model_robust_s6.p')
    intermediate_folder: Union[str, Path] = Path('/n/groups/datta/Dana/Ontogeny/dataframes/step_01')
