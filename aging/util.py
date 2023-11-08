import pandas as pd
from typing import Union
from sklearn.decomposition import PCA
from toolz import frequencies, valfilter, sliding_window

def zscore(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Compute z-score of a dataset.

    Parameters
    ----------
    x: pd.Series or pd.DataFrame
        array of values.

    Returns
    -------
    z: pd.Series or pd.DataFrame 
        array of z-scored values.
    """
    return (x - x.mean()) / x.std()


def minmax_normalize(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Compute min-max normalization of a dataset.

    Parameters
    ----------
    x: pd.Series or pd.DataFrame
        array of values.

    Returns
    -------
    z: pd.Series or pd.DataFrame 
        array of min-max normalized values.
    """
    return (x - x.min()) / (x.max() - x.min())


def pca(data, n_components=None):
    """Perform PCA on a dataset."""
    _pca = PCA(n_components=n_components)
    out = _pca.fit_transform(data)
    return out, _pca


def count_ngram(sequence, n=2, threshold_counts=0):
    '''
    Counts the number of times an ngram appears in a sequence.
    '''
    return valfilter(lambda x: x > threshold_counts, frequencies(sliding_window(n, sequence)))


def count_trigram(sequence, threshold_counts=0):
    return count_ngram(sequence, n=3, threshold_counts=threshold_counts)


def bigram_normalize(tm):
    return tm / tm.sum()


def row_normalize(tm):
    '''Computes outgoing transition probabilities.'''
    return tm / tm.sum(axis=1, keepdims=True)


def correct_for_camera_height(array, camera_height, target_height=670):
    '''Corrects for camera height.'''
    return array * (camera_height / target_height)