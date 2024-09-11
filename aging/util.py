import h5py
import numpy as np
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


def copy_attributes(source, destination):
    """Copy attributes from source object to destination object"""
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def recursive_copy(source_group, destination_group):
    """Recursively copy groups and datasets from source to destination"""
    for name, item in source_group.items():
        if isinstance(item, h5py.Group):
            # Create group in destination
            new_group = destination_group.create_group(name)
            # Copy attributes
            copy_attributes(item, new_group)
            # Recursive call for nested groups
            recursive_copy(item, new_group)
        elif isinstance(item, h5py.Dataset):
            # Create dataset in destination with compression if not scalar
            if item.shape != () and item.shape is not None:  # Check if dataset is not a scalar
                chunks = True  # Let h5py decide chunk size
                new_dataset = destination_group.create_dataset(
                    name,
                    data=item[:],
                    chunks=chunks,
                    compression="gzip",
                    compression_opts=4,  # Compression level (1-9, 9 is highest)
                )
            else:
                # For scalar datasets, create without compression
                new_dataset = destination_group.create_dataset(name, data=item[()])

            # Copy attributes
            copy_attributes(item, new_dataset)


def copy_h5_file(source_path, destination_path):
    """Copy entire HDF5 file from source to destination"""
    with h5py.File(source_path, "r") as source_file, h5py.File(
        destination_path, "w"
    ) as destination_file:
        # Copy root level attributes
        copy_attributes(source_file, destination_file)
        # Start recursive copy
        recursive_copy(source_file, destination_file)


def compare_h5_files(file1_path, file2_path):
    """
    Compare two HDF5 files to ensure they contain the same information.
    Returns True if files are identical, False otherwise.
    """
    with h5py.File(file1_path, 'r') as file1, h5py.File(file2_path, 'r') as file2:
        return compare_groups(file1, file2)


def compare_groups(group1, group2):
    """Recursively compare groups and datasets"""
    if set(group1.keys()) != set(group2.keys()):
        print(f"Group keys don't match: {group1.name}")
        return False

    if not compare_attributes(group1, group2):
        print(f"Attributes don't match for group: {group1.name}")
        return False

    for key in group1.keys():
        if isinstance(group1[key], h5py.Group):
            if not isinstance(group2[key], h5py.Group):
                print(f"Item type mismatch: {key}")
                return False
            if not compare_groups(group1[key], group2[key]):
                return False
        elif isinstance(group1[key], h5py.Dataset):
            if not isinstance(group2[key], h5py.Dataset):
                print(f"Item type mismatch: {key}")
                return False
            if not compare_datasets(group1[key], group2[key]):
                return False

    return True


def compare_datasets(dataset1, dataset2):
    """Compare two datasets"""
    if dataset1.shape != dataset2.shape:
        print(f"Shape mismatch for dataset: {dataset1.name}")
        return False

    if dataset1.dtype != dataset2.dtype:
        print(f"Dtype mismatch for dataset: {dataset1.name}")
        return False

    float_flag = dataset1.dtype == np.float32 or dataset1.dtype == np.float64
    if not np.array_equal(dataset1[()], dataset2[()], equal_nan=float_flag):
        print(f"Data mismatch for dataset: {dataset1.name}")
        return False

    if not compare_attributes(dataset1, dataset2):
        print(f"Attributes don't match for dataset: {dataset1.name}")
        return False

    return True


def compare_attributes(obj1, obj2):
    """Compare attributes of two objects"""
    if set(obj1.attrs.keys()) != set(obj2.attrs.keys()):
        return False

    for key in obj1.attrs.keys():
        if not np.array_equal(obj1.attrs[key], obj2.attrs[key]):
            return False

    return True