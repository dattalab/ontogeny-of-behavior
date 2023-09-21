import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from toolz import pipe
from tqdm.auto import tqdm
from typing import Callable, Iterable, Optional
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform, pdist
from scipy.stats import rankdata
from aging.plotting import figure


def zscore(df):
    return (df - df.mean()) / df.std()


def pca(df, n_dims=10):
    return pd.DataFrame(PCA(n_dims).fit_transform(df), index=df.index)


def compute_nearest_neighbors(df, metric, k_neigh):
    distances = squareform(pdist(df.to_numpy(), metric=metric))
    nearest = np.argsort(distances, axis=1)
    nn = nearest[:, : k_neigh + 1]
    return nn, distances[nn[:, [0]], nn]


def minmax_norm(x):
    return (x - x.min()) / np.ptp(x)


def preprocess_df(usage_df, filter_fun=None, xform_fun=None):
    if filter_fun is not None:
        if not isinstance(filter_fun, Iterable):
            filter_fun = [filter_fun]
        usage_df = pipe(usage_df, *filter_fun)
    if xform_fun is not None:
        usage_df = xform_fun(usage_df)
    return usage_df


def diffuse_pseudotime(seed_vec: np.ndarray, smoothing_mtx: np.ndarray, diffusion_iter: int = 5_000) -> tuple[np.ndarray, np.ndarray]:
    out = seed_vec.copy()
    for _ in tqdm(range(diffusion_iter)):
        out = minmax_norm(smoothing_mtx @ out)

    ranks = rankdata(1 - out)
    ranks = ranks / ranks.max()
    return ranks, out


def make_smoothing_mtx(neighhors):
    smoothing_mtx = np.zeros((len(neighbors),) * 2)
    smoothing_mtx[neighbors[:, [0]], neighbors] = 1 / k_neigh
    smoothing_mtx = np.eye(len(neighbors)) * beta * smoothing_mtx + (1 - beta) * smoothing_mtx
    return smoothing_mtx


def compute_pseudotime(
    usage_df: pd.DataFrame,
    filter_fun: Optional[Callable | Iterable[Callable]] = None,
    xform_fun: Optional[Callable] = None,
    metric: str | Callable = "euclidean",
    k_neigh: int = 7,
    beta: float = 0.1,
    diffusion_iter: int = 5_000,
) -> pd.DataFrame:
    """Computes a measure of pseudotime based on nearest-neighbor similarities
    Parameters:
        usage_df (pd.DataFrame): dataframe of normalized syllable frequencies, where each row is a different session
        filter_fun (function or list of functions): function(s) that take in the usage_df as the first parameter
            and return a dataframe, removing some rows if they match some exclusion criteria defined in the
            filter function. Example: removing infrequent syllables.
        xform_fun (function): function to be applied to the usage_df to produce some sort of normalization.
            Example: zscoring syllable usages across sessions.
        metric (str or function): distance metric to compare syllable usage across sessions
        beta (float): off-diagonal weighted sum magnitude
    Returns:
        pseudotime (pd.DataFrame): pseudotime ranks and smoothed values with the same usage_df index
    """
    usage_df = preprocess_df(usage_df, filter_fun, xform_fun)

    nn, _ = compute_nearest_neighbors(usage_df, metric, k_neigh)

    smoothing_mtx = make_smoothing_mtx(nn)

    seed_idx = np.where(
        usage_df.index.get_level_values("age")
        == usage_df.index.get_level_values("age").min()
    )[0]
    pseudo_vals = np.zeros(len(nn))
    pseudo_vals[seed_idx] = 1
    ranks, out = diffuse_pseudotime(pseudo_vals, smoothing_mtx, diffusion_iter)

    return pd.DataFrame(
        dict(pseudotime_rank=ranks, pseudotime_dist=out), index=usage_df.index
    )


def pseudotime_springplot(
    usage_df: pd.DataFrame,
    filter_fun: Optional[Callable | Iterable[Callable]] = None,
    xform_fun: Optional[Callable] = None,
    metric: str | Callable = "euclidean",
    k_neigh: int = 7,
    seed: int = 0,
    cmap: str = "mako",
    node_size: int = 4,
    width: int = 1,
    height: int = 1,
    **fig_kwds,
) -> tuple[nx.Graph, dict]:
    usage_df = preprocess_df(usage_df, filter_fun, xform_fun)
    ages = usage_df.index.get_level_values("age").to_numpy()
    nn, dist = compute_nearest_neighbors(usage_df, metric, k_neigh)

    graph = nx.Graph()
    edges = []
    for row in nn:
        i = row[0]
        for k, j in enumerate(row[1:], start=1):
            edges.append((i, j, dist[i, k]))
    graph.add_weighted_edges_from(edges)
    colors = [ages[i] for i in graph.nodes]
    pos = nx.spring_layout(graph, seed=seed)

    fig = figure(width, height, **fig_kwds)
    ax = fig.gca()
    nx.draw(
        graph,
        pos=pos,
        node_size=node_size,
        width=0.1,
        edge_color="gray",
        node_color=colors,
        arrows=False,
        cmap=cmap,
        ax=ax,
    )
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=ages.min(), vmax=ages.max())
    )
    fig.colorbar(sm, ax=ax, label="Age")
    return graph, pos, colors
