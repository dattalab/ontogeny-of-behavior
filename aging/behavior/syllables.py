import pandas as pd

def compute_onsets(df: pd.DataFrame):
    mask = (df['syllables'].diff() != 0).fillna(False).astype(bool)
    return mask


def assign_onsets(df: pd.DataFrame):
    df['onsets'] = compute_onsets(df)
    return df


def compute_usage(df):
    if 'onsets' not in df.columns:
        df = assign_onsets(df)
    usage = df.loc[df['onsets'], 'syllables'].value_counts()
    return usage


def relabel_by_usage(df: pd.DataFrame, return_map=True):
    # old -> new
    usage_map = dict(map(reversed, enumerate(compute_usage(df).index)))
    df['relabeled_syllables'] = df['syllables'].map(usage_map)
    if return_map:
        return df, usage_map
    return df
    