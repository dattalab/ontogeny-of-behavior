'''
Defines a series of functions useful for plotting figures.
'''
import platform
import matplotlib.pyplot as plt
import seaborn as sns
from toolz import merge
from pathlib import Path
from dataclasses import dataclass

IMG_KWARGS = dict(aspect='auto', interpolation='none')

ont_male_colors = ['#c7eae5', '#008C8D']
ont_female_colors = ['#fee6ce', '#d94801']
long_male_colors = ['#DADAEB', '#6A51A3']

@dataclass
class Colormaps:
    ont_male = sns.blend_palette(ont_male_colors, as_cmap=True)
    ont_female = sns.blend_palette(ont_female_colors, as_cmap=True)
    long_male = sns.blend_palette(long_male_colors, as_cmap=True)


COLORMAPS = Colormaps()


@dataclass
class PlotConfig:
    save_path: Path = Path("/n/groups/datta/win/figures/ontogeny")
    dana_save_path: Path = Path("/n/groups/datta/Dana/Ontogeny/figs")


PLOT_CONFIG = PlotConfig()


def figure(width, height, dpi=300, **kwargs):
    return plt.figure(figsize=(width, height), dpi=dpi, **kwargs)


def legend(ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False, **kwargs)


def format_plots():
    '''
    Defines a series of formatting options for plots and applies them globally.
    '''
    all_fig_dct = {
        "pdf.fonttype": 42,
        "figure.figsize": (3, 3),
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        'savefig.facecolor': 'white',
        'savefig.transparent': True,
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        "axes.labelcolor": "black",
        "text.color": "black",
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none',
        'lines.linewidth': 1,
        'axes.linewidth': 0.5,
        "axes.unicode_minus": False,
    }

    # all in points
    font_dct = {
        "axes.labelpad": 2.5,
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "xtick.major.size": 1.75,
        "ytick.major.size": 1.75,
        "xtick.minor.size": 1.75,
        "ytick.minor.size": 1.75,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "xtick.minor.pad": 1,
        "ytick.minor.pad": 1,
    }

    plot_config = merge(all_fig_dct, font_dct)

    plt.style.use('default')
    for k, v in plot_config.items():
        plt.rcParams[k] = v
    sns.set_style('white', merge(sns.axes_style('ticks'), plot_config))
    sns.set_context('paper', rc=plot_config)

    if platform.system() != 'Darwin':
        plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 200


def save_factory(folder, backgrounds=('white',), tight_layout=True, dpi=200):
    folder = Path(folder).absolute().expanduser()
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, name, savefig=True, tight_layout=tight_layout, dpi=dpi):
        if tight_layout:
            fig.tight_layout()
        if savefig:
            for bg in backgrounds:
                ext = '' if len(backgrounds) == 1 else f'_{bg}'
                for ax in fig.axes:
                    ax.set_facecolor(bg)
                fig.savefig(folder / (name + ext + '.png'),
                            dpi=dpi, facecolor=bg)
                fig.savefig(folder / (name + ext + '.pdf'), facecolor=bg, dpi=dpi)
        return fig

    return save


def format_pizza_plots():
    '''
    Defines a series of formatting options for plots and applies them globally.
    '''
    all_fig_dct = {
        "pdf.fonttype": 42,
        "figure.figsize": (3, 3),
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        'savefig.facecolor': 'black',
        # 'savefig.transparent': True,
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        "axes.labelcolor": "white",
        "text.color": "white",
        'xtick.color': 'white',
        'ytick.color': 'white',
        'svg.fonttype': 'none',
        'lines.linewidth': 1,
        'axes.linewidth': 1,
    }

    # all in points
    font_dct = {
        "axes.labelpad": 2.5,
        "font.size": 6,
        "axes.titlesize": 8,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "xtick.major.size": 1.75,
        "ytick.major.size": 1.75,
        "xtick.minor.size": 1.75,
        "ytick.minor.size": 1.75,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "xtick.minor.pad": 1,
        "ytick.minor.pad": 1,
    }

    plot_config = merge(all_fig_dct, font_dct)

    plt.style.use('default')
    for k, v in plot_config.items():
        plt.rcParams[k] = v
    sns.set_style('dark', merge(sns.axes_style('ticks'), plot_config))
    sns.set_context('paper', rc=plot_config)

    if platform.system() != 'Darwin':
        plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 200


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes