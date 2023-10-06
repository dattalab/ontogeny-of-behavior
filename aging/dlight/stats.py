import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange


def bootstrap_lineplot(
    x, y, ci=99, label=None, color=None, alpha=0.5, mu=None, linewidth=1, ax=None, **plt_kwargs
):
    '''Expects a tidy dataframe that has already been bootstrapped. I.e., each trial is the
    average of a different sample (with replacement) of the original data'''
    if mu is None:
        _mu = np.nanmean(y, axis=0)
        mu = _mu
    if ci == "sd":
        lo = mu - 2 * np.nanstd(y, axis=0)
        hi = mu + 2 * np.nanstd(y, axis=0)
    else:
        lo = np.nanquantile(y, 1 - (ci / 100), axis=0)
        hi = np.nanquantile(y, ci / 100, axis=0)

    plotter = plt if ax is None else ax

    lines = plotter.plot(x, mu, label=label, color=color, linewidth=linewidth, **plt_kwargs)
    color = lines[-1].get_color()
    plotter.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, **plt_kwargs)

    return plt.gca() if ax is None else ax


@jit(nopython=True, parallel=True)
def bootstrap_ci(data, n_boots=1000):
    '''data = M x N matrix, where M is the number of trials, N is the number of observations
    Returns:
        mus: n_boots x N matrix'''
    mus = np.zeros((n_boots, data.shape[1]))
    n_trials = len(data)
    for i in prange(n_boots):
        choices = np.random.choice(np.arange(n_trials), n_trials, replace=True)
        arr = np.zeros_like(data)
        for j, c in enumerate(choices):
            arr[j] = data[c]
        for j in range(data.shape[1]):
            mus[i, j] = np.nanmean(arr[:, j])
    return mus
