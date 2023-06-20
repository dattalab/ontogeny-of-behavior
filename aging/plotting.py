'''
Defines a series of functions useful for plotting figures.
'''
import platform
import matplotlib.pyplot as plt
import seaborn as sns
from toolz import merge
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PlotConfig:
    save_path: Path = Path("/n/groups/datta/win/figures/ontogeny")


def format_plots():
    '''
    Defines a series of formatting options for plots and applies them globally.
    '''
    all_fig_dct = {
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        # "font.sans-serif": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        "axes.labelcolor": "black",
        "text.color": "black",
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none',
        'lines.linewidth': 1,
    }

    # all in points
    font_dct = {
        "axes.labelpad": 3.5,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5
    }

    plot_config = merge(all_fig_dct, font_dct)

    plt.style.use('default')
    sns.set_style('white', merge(sns.axes_style('ticks'), plot_config))
    sns.set_context('paper', rc=plot_config)

    if platform.system() != 'Darwin':
        plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 200


def save_factory(folder, backgrounds=('white',), tight_layout=True):
    folder = Path(folder).absolute().expanduser()
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, name, savefig=True, tight_layout=tight_layout):
        if tight_layout:
            fig.tight_layout()
        if savefig:
            for bg in backgrounds:
                ext = '' if len(backgrounds) == 1 else f'_{bg}'
                for ax in fig.axes:
                    ax.set_facecolor(bg)
                fig.savefig(folder / (name + ext + '.png'),
                            dpi=150, facecolor=bg)
                fig.savefig(folder / (name + ext + '.pdf'), facecolor=bg)
        return fig

    return save