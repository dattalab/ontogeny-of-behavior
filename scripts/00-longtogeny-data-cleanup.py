'''
This script parses and organizes data from the longtogeny experiments.
This script requires moseq2-viz to be installed.
'''
from aging.organization.longtogeny import LongtogenyPaths
from moseq2_viz.model.util import parse_model_results
from moseq2_viz.util import parse_index
from toolz import groupby, assoc, curry, valmap, valfilter
from tqdm.auto import tqdm
from pathlib import Path
import re
import datetime
import numpy as np
import pandas as pd


def parse_date(date_str):
    return datetime.datetime.strptime(date_str, 'session_%Y%m%d%H%M%S')


def make_new_subject_name(d: dict):
    '''Make a new subject name from the session name and subject name and assign value to 'subject_name' key in dict.'''
    return assoc(d, 'subject_name', '_'.join([d['session_name'][:20], d['subject_name'][:20]]))


@curry
def filter_session(rec: str, d: dict) -> bool:
    '''Filter out sessions that aren't part of this experiment'''
    # note: this filter also removes "nor" (novel object recognition) sessions
    #  and ADD (alzheimer's disease) sessions
    if rec == 'rec_lon_male':
        # remove any session with "default" in name
        if 'default' in d['subject_name'].lower():
            return False
        # remove any session with "m" in name - like "c02_m08" or "C01_m05"
        if 'm' in d['subject_name'].lower():
            return False
    return True
    

def fix_ontogeny_ages(d: pd.Series) -> str:
    '''Compute age mapping for ontogeny dataset'''
    # search for specific string pattern in ontogeny dataset
    name_mapping = {
        '3_weeks': re.compile(r'(y_03)|(_3wk_)'),
        '5_weeks': re.compile(r'(y_05)|(_5wk_)'),
        '7_weeks': re.compile(r'(y_07)|(_7wk_)'),
        '9_weeks': re.compile(r'(y_09)|(_9wk_)'),
        '3_months': re.compile(r'3m_o'),
        '6_months': re.compile(r'6m_o'),
        '12_months': re.compile(r'_12m_'),
        '18_months': re.compile(r'_18m_'),
    }
    output = valmap(lambda v: v.search(d['session_name'][:20] + '_' + d['subject_name'][:20]), name_mapping)
    output = valfilter(lambda v: v is not None, output)
    assert len(output) == 1, 'Could not find age for session: {}'.format(d['session_name'])

    return list(output)[0]


def main():
    paths = LongtogenyPaths()
    
    ### section 0 ###
    print('Loading in model results from', paths.model_file)
    # parse data using original moseq way
    model_results = parse_model_results(str(paths.model_file), sort_labels_by_usage=True)
    _, sorted_index = parse_index(str(paths.index_file))
    
    # assumes model_results['labels'] is a list and not already a dict
    labels = dict(zip(model_results['keys'], model_results['labels']))
    
    ### section 1 ###
    print('Looping through files, extracting metadata. # files:', len(sorted_index['files']))
    files = []
    # assumes sorted_index['files'] is dict
    for uuid, v in sorted_index['files'].items():
        pth = Path(v['path'][0])
        files.append(dict(
            uuid=uuid,
            subject_name=v['metadata']['SubjectName'],
            session_name=v['metadata']['SessionName'],
            session_date=parse_date(pth.parents[1].name),
            session_date_name=pth.parents[1].name,
            start_time=v['metadata'].get('StartTime'),
            group=v['group'],
            rec_full=pth.parents[2].name,
            rec='_'.join(pth.parents[2].name.split('_')[:-1]),
            full_path=str(pth),
        ))
        
    ### section 2 ###
    # having a harder time understanding the point of this code snippet
    #   - probably need to run it to find out
    print('Organize files by experimental group')
    grouped_files = groupby('rec', files)

    # update subject names for this experimental group
    # grouped_files['rec_ont_male'] = [make_new_subject_name(d) for d in grouped_files['rec_ont_male']]

    ### section 3 ###
    # create dataframe for each experimental group

    dtype_map = {
        'uuid': 'category',
        'subject_name': 'category',
        'session_name': 'category',
        'session_date_name': 'category',
        'start_time': 'category',
        'group': 'category',
        'rec_full': 'category',
        'rec': 'category',
        'full_path': 'category',
        'labels': 'int16[pyarrow]',
    }

    print('Loop through each experimental group and create dataframe')
    for rec, data in grouped_files.items():
        experiment_df = []

        # loop through each session in this experimental group, filter bad sessions
        for d in tqdm(filter(filter_session(rec), data), desc=rec):
            if d['uuid'] in labels:
                # add syllable labels to the metadata dictionary
                d['labels'] = labels[d['uuid']]
                d['onset'] = np.concatenate(([0], np.diff(d['labels'].astype(int)) != 0)).astype(bool)
                if rec == 'rec_ont_male':
                    d['age'] = fix_ontogeny_ages(d)

                # aggregate
                experiment_df.append(pd.DataFrame(d).astype(dtype_map))
        # concat all sessions into one dataframe for this experimental group
        experiment_df = pd.concat(experiment_df, ignore_index=True).astype(dtype_map)

        if rec != 'rec_ont_male':
            mouse_id = experiment_df['subject_name'].str[:2].astype('int8')
            cage_id = experiment_df['subject_name'].str[3:5].astype('int8')

            # compute age from the session recording date
            earliest_mouse_age = 24  # days
            earliest_date = experiment_df['session_date'].min()
            age = (experiment_df['session_date'] - earliest_date).dt.days + earliest_mouse_age

            experiment_df = experiment_df.assign(mouse_id=mouse_id, cage_id=cage_id, age=age)
        elif rec == "rec_ont_male":
            experiment_df = experiment_df.astype(dict(age="category"))

        # TODO: handle spelling errors, ontogeny exp naming
        experiment_df.to_parquet(paths.intermediate_folder / f'{rec}.parquet', index=False)


if __name__ == '__main__':
    main()