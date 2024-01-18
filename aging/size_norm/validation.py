import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from shutil import copyfile
from aging.size_norm.util import multi_stage_pca, flatten
from sklearn.metrics import confusion_matrix, accuracy_score
from toolz import partial, compose, valmap, itemmap, keyfilter, merge
from aging.plotting import format_plots, figure, save_factory, legend, IMG_KWARGS
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold


### ~~~ age classification from raw or size-normalized images ~~~ ###


def transformed_file_exists(model_path: Path):
    return (
        model_path.parent
        / "age_classification"
        / "transformed_age_classification_poses.p.gz"
    ).exists()


def _transform_data(data: dict, model_path: Path | str):
    from aging.size_norm.data import Session
    from aging.size_norm.lightning import predict

    def clean_img(img):
        img = img.copy()
        img[img < 10] = 0
        return img

    predict_fun = compose(
        np.uint8,
        np.round,
        partial(predict, model=model_path, disable=True),
        Session,
        clean_img,
    )
    return valmap(predict_fun, data)


def transform_data(
    data_path: Path, model_path: Path | str, data_source: str = "age_classification"
):
    if isinstance(model_path, str):
        model_path = Path(model_path)

    save_path = model_path.parent / data_source
    save_path.mkdir(exist_ok=True)
    save_file = save_path / f"transformed_{data_source}_poses.p.gz"

    # short circuit if already exists
    if save_file.exists():
        return save_file

    # load data
    data = joblib.load(data_path)
    data = _transform_data(data, model_path)

    joblib.dump(data, save_file, compress=3)
    return save_file


# perform age classification
def age_classifiers(data_path: Path, debug: bool = False, thinning: int = 1):
    """data_path must point to SNN transformed data"""
    from cuml.svm import LinearSVC
    from cuml.preprocessing import StandardScaler
    from cuml.ensemble import RandomForestClassifier
    from cuml.pipeline import make_pipeline

    if debug:
        print("Loading data")
    data = joblib.load(data_path)
    if debug:
        print("Done loading data")

    # has the same dict structure as data
    if debug:
        print("Running pca")
    pcs = multi_stage_pca(data)
    if debug:
        print("Done with pca")
    X = np.concatenate(list(pcs.values()), axis=0).astype("float32")
    Y = np.concatenate(list(map(lambda x: np.repeat(x[0][0], len(x[1])), data.items())))

    if thinning > 1:
        X = X[::thinning]
        Y = Y[::thinning]

    gridsearch_partial = partial(
        GridSearchCV,
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
    )

    if debug:
        print("Running SVC gridsearch")

    # run SVC gridsearch
    pipeline = make_pipeline(StandardScaler(), LinearSVC())
    grid = gridsearch_partial(
        pipeline,
        param_grid={
            "linearsvc__C": [0.1, 0.5, 1, 5, 10],
        },
    )
    grid.fit(X, Y)

    if debug:
        print("Done with SVC gridsearch")

    cms = []
    accs = []
    for i in range(10):
        predictions = cross_val_predict(grid.best_estimator_, X, Y, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=i))
        _cm = confusion_matrix(Y, predictions, normalize="pred")
        cms.append(_cm)
        accs.append(accuracy_score(Y, predictions))

    svc_cm = np.mean(cms, axis=0)
    svc_acc = np.mean(accs)

    del grid

    if debug:
        print("Running RF gridsearch")

    # run RF gridsearch
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    grid = gridsearch_partial(
        pipeline,
        param_grid={
            "randomforestclassifier__n_estimators": [10, 50, 100, 200],
        },
    )
    grid.fit(X, Y)

    if debug:
        print("Done with RF gridsearch")

    cms = []
    accs = []
    for i in range(10):
        predictions = cross_val_predict(grid.best_estimator_, X, Y, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=i))
        _cm = confusion_matrix(Y, predictions, normalize="pred")
        cms.append(_cm)
        accs.append(accuracy_score(Y, predictions))

    rf_cm = np.mean(cms, axis=0)
    rf_acc = np.mean(accs)

    del grid

    return {
        "svc": {
            "cm": svc_cm,
            "acc": svc_acc,
            "chance": 1 / len(np.unique(Y)),
        },
        "rf": {
            "cm": rf_cm,
            "acc": rf_acc,
            "chance": 1 / len(np.unique(Y)),
        },
    }


# plot results
def plot_classification(
    classifier_results: dict, save_path: Path, is_transformed: bool
):
    format_plots()
    saver = save_factory(save_path, tight_layout=True)

    # plot svc confusion matrix
    fig = figure(2.4, 2)
    ax = sns.heatmap(
        classifier_results["svc"]["cm"],
        annot=False,
        cmap="cubehelix",
        cbar_kws={"label": "Proportion"},
        yticklabels=False,
        xticklabels=False,
        vmin=0,
        vmax=0.5,
    )
    ax.set(
        xlabel="Predicted Age (wks)",
        ylabel="True Age (wks)",
        aspect="equal",
        title="SVC Age Classification",
    )
    saver(fig, "svc_confusion_matrix" if is_transformed else "svc_confusion_matrix_og")

    # plot rf confusion matrix
    fig = figure(2.4, 2)
    ax = sns.heatmap(
        classifier_results["rf"]["cm"],
        annot=False,
        cmap="cubehelix",
        cbar_kws={"label": "Proportion"},
        yticklabels=False,
        xticklabels=False,
        vmin=0,
        vmax=0.5,
    )
    ax.set(
        xlabel="Predicted Age (wks)",
        ylabel="True Age (wks)",
        aspect="equal",
        title="RF Age Classification",
    )
    saver(fig, "rf_confusion_matrix" if is_transformed else "rf_confusion_matrix_og")

    # write accuracy to json file
    name = "accuracy.txt" if is_transformed else "accuracy_og.txt"
    with open(save_path / name, "w") as f:
        json.dump(
            {
                "svc": classifier_results["svc"]["acc"],
                "rf": classifier_results["rf"]["acc"],
                "chance": classifier_results["rf"]["chance"],
            },
            f,
            indent=4,
        )


# save results to separate folder, symlink to model folder
def link_results_folder(saved_path: Path, model_folder: Path):
    # link saved_path to a folder inside model_path
    link_path = model_folder / saved_path.name
    if link_path.exists():
        link_path.unlink()
    link_path.symlink_to(saved_path)

    # try to copy the config file over to the saved_path
    config_path = model_folder / "config.toml"
    if config_path.exists():
        copyfile(config_path, saved_path / "config.toml")


### ~~~ visualization of pose space manifold ~~~ ###


def pca_pose_manifold(pre_xform_path: Path, post_xform_path: Path, save_path: Path):
    data = joblib.load(pre_xform_path)
    # data must be a dict - assuming it's the same data as used for age classification
    pcs = multi_stage_pca(data, subset_frames=400)

    plot_manifold(pcs, save_path, suffix="pca_pre_xform")

    data = joblib.load(post_xform_path)
    pcs = multi_stage_pca(data, subset_frames=400)
    plot_manifold(pcs, save_path, suffix="pca_post_xform")

    return save_path


# use for both umap and pca manifolds
def plot_manifold(
    data: dict, save_path: Path, suffix: str = "pca", thinning: int = 5, seed: int = 0
):
    format_plots()
    rng = np.random.default_rng(seed)
    saver = save_factory(save_path, tight_layout=True)

    data = valmap(lambda v: v[::thinning], data)
    colors = {k: [k[0]] * len(v) for k, v in data.items()}

    data = np.vstack(list(data.values()))

    permutation = rng.permutation(len(data))
    data = data[permutation]
    colors = np.concatenate(list(colors.values()))[permutation]

    fig = figure(1.30, 1.15)
    ax = fig.gca()
    im = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=colors,
        cmap="cubehelix",
        s=1,
        lw=0.05,
        ec="k",
        alpha=0.9,
        rasterized=True,
    )
    fig.colorbar(im, ax=ax, label="Age (weeks)")
    ax.set(
        xticks=[],
        yticks=[],
        xlabel="Dim 1",
        ylabel="Dim 2",
    )
    sns.despine()
    saver(fig, f"pose_manifold_{suffix}", dpi=600)


### ~~~ changepoints correlation ~~~ ###


def zscore(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / arr.std(axis=0, keepdims=True)


def _alignment_function(frame, recon_frame):
    from scipy.ndimage import gaussian_filter, rotate, shift
    from aging.size_norm.data import clean
    from aging.behavior.scalars import im_moment_features

    super_clean = partial(clean, height_thresh=14, tail_ksize=7, dilation_ksize=4)

    frame = super_clean(frame)
    filt_frame = gaussian_filter(frame.astype("float32"), 13)
    filt_recon_frame = gaussian_filter(recon_frame.astype("float32"), 13)

    raw_peak = np.array(np.unravel_index(np.argmax(filt_frame), filt_frame.shape))
    recon_peak = np.array(
        np.unravel_index(np.argmax(filt_recon_frame), filt_recon_frame.shape)
    )
    translated = shift(frame, recon_peak - raw_peak)

    moments = im_moment_features(translated)
    if moments is None:
        return frame
    rotated = rotate(
        translated, -np.rad2deg(moments["orientation"]), reshape=False
    ).astype("uint8")
    return rotated

def compute_alignments(og_frames, xform_frames):
    return np.array([_alignment_function(f, r) for f, r in zip(og_frames, xform_frames)])

def _compute_changepoints(frames, mdl=None, k=5, sig=3):
    from aging.size_norm.data import clean

    if mdl is None:
        from sklearn.random_projection import GaussianRandomProjection

        mdl = GaussianRandomProjection(n_components=300, random_state=0)

        frames = np.array([clean(f) for f in frames])
        proj = mdl.fit_transform(flatten(frames))
    else:
        proj = mdl.transform(flatten(frames))

    proj_df = pd.DataFrame(zscore(zscore(proj).T).T, index=np.arange(len(proj)) / 30)
    proj_df_smooth = (
        proj_df.rolling(sig * 4, win_type="gaussian", center=True)
        .mean(std=sig)
        .dropna()
    )
    squared_diff = np.square(proj_df_smooth.diff(k)).shift(-k // 2)
    cp = squared_diff.mean(axis="columns")

    return cp, proj_df_smooth, mdl


def compute_changepoints(pre_xform_path: Path, post_xform_path: Path):
    pre_xform = joblib.load(pre_xform_path)
    post_xform = joblib.load(post_xform_path)

    pre_xform = {k: compute_alignments(v, post_xform[k]) for k, v in pre_xform.items()}

    pre_cps = valmap(_compute_changepoints, pre_xform)
    post_cps = itemmap(
        lambda args: (args[0], _compute_changepoints(args[1], mdl=pre_cps[args[0]][2])),
        post_xform,
    )

    return pre_cps, post_cps


def _imshow(ax, data, cmap, lims):
    ax.imshow(
        data,
        cmap=cmap,
        vmin=-lims,
        vmax=lims,
        extent=(0, data.shape[1] / 30, 0, len(data)),
        **IMG_KWARGS,
    )
    ax.set(yticks=[], xlabel="Time (s)", ylabel="Random projections")
    return ax


def _plot_changepoints(pre, post, cmap="RdBu_r", lims=1.5, start_idx=0, n_frames=750):
    import matplotlib.pyplot as plt

    pre_cp, pre_df = pre
    post_cp, post_df = post

    fig, ax = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [1, 1, 1.4]}, figsize=(3, 3), sharex=True
    )

    _imshow(ax[0], pre_df.iloc[start_idx : start_idx + n_frames].T, cmap, lims)
    _imshow(ax[1], post_df.iloc[start_idx : start_idx + n_frames].T, cmap, lims)

    ax[2].plot(
        np.arange(n_frames) / 30,
        post_cp.iloc[start_idx : start_idx + n_frames],
        color="k",
        label="Transformed",
    )
    ax[2].plot(
        np.arange(n_frames) / 30,
        pre_cp.iloc[start_idx : start_idx + n_frames],
        color="silver",
        label="Original",
    )
    legend(ax[2])
    ax[2].set(ylabel="Changepoint score (a.u.)", xlabel="Time (s)", xlim=(0, 25))
    sns.despine(ax=ax[2])
    return fig


def plot_changepoints(
    pre_cps: dict,
    post_cps: dict,
    save_path: Path,
    cmap="RdBu_r",
    lims=1.5,
    start_idx=0,
    n_frames=750,
):
    format_plots()
    save_path = save_path / "session_changepoints"
    saver = save_factory(save_path, tight_layout=True)
    _plt_fun = partial(
        _plot_changepoints, cmap=cmap, lims=lims, start_idx=start_idx, n_frames=n_frames
    )

    for key in pre_cps:
        filename = f"changepoints_{key[1].parents[1].name}"
        fig = _plt_fun(pre_cps[key][:2], post_cps[key][:2])
        fig.suptitle(f"{key[0]} wks")
        saver(fig, filename)
    return save_path


def plot_changepoint_correlations(pre_cps: dict, post_cps: dict, save_path: Path):
    from aging.plotting import ONTOGENY_AGE_CMAP

    format_plots()

    saver = save_factory(save_path, tight_layout=True)

    corrs = []
    for key in pre_cps:
        corr = pre_cps[key][0].corr(post_cps[key][0])
        corrs.append(dict(corr=corr, age=key[0]))
    corrs = pd.DataFrame(corrs)

    fig = figure(2, 2)
    ax = sns.scatterplot(
        data=corrs, x="age", y="corr", palette=ONTOGENY_AGE_CMAP, s=5, hue="age"
    )
    legend(ax, title="Age (weeks)")
    ax.set(ylim=(0, 1), ylabel="Changepoint score correlation")
    sns.despine()

    saver(fig, "cps-correlation-sscatter")
    with open(save_path / "cps-correlation.txt", "w") as f:
        json.dump({"correlation": corrs["corr"].median()}, f)
    return save_path


### ~~~ dynamics correlation ~~~ ###
def zip_valmap(func, *ds):
    new_dict = {}
    for k in ds[0]:
        new_dict[k] = func(*(d[k] for d in ds))
    return new_dict


def _compute_dynamics(og_frames, xform_frames):
    from aging.behavior.scalars import compute_scalars

    scalars = compute_scalars(og_frames, is_recon=False, height_thresh=10, clean_flag=True)
    xform_scalars = compute_scalars(xform_frames, is_recon=True, height_thresh=10, clean_flag=True)

    return merge(scalars, xform_scalars)


def compute_dynamics(pre_xform_path: Path, post_xform_path: Path):
    pre_xform = joblib.load(pre_xform_path)
    post_xform = joblib.load(post_xform_path)

    scalars = zip_valmap(_compute_dynamics, pre_xform, post_xform)

    return scalars


def _plot_dynamics(df: pd.DataFrame, age: int, n_frames: int = 750):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(4, 3), sharex=True)
    for i, col in enumerate(filter(lambda c: 'recon' not in c, df.columns)):
        ax[i].plot(np.arange(n_frames) / 30, df[col].iloc[:n_frames], color="k", label="Original")
        ax[i].plot(np.arange(n_frames) / 30, df[f"recon_{col}"].iloc[:n_frames], color="silver", label="Transformed")
        ax[i].set(ylabel=col)
    ax[-1].set(xlabel="Time (s)", title=f"{age} wks")
    legend(ax[-1])
    sns.despine()
    return fig


def zscore_pd(df: pd.DataFrame):
    return (df - df.mean()) / df.std()


def plot_dynamics(dynamics_data: dict, save_path: Path, n_frames: int = 750):
    from aging.plotting import ONTOGENY_AGE_CMAP
    format_plots()
    saver = save_factory(save_path, tight_layout=True)
    
    corrs = []
    for k, v in dynamics_data.items():
        tmp_df = pd.DataFrame(v)
        c = [tmp_df[k].corr(tmp_df[f"recon_{k}"]) for k in tmp_df.columns if 'recon' not in k]
        corrs.append(dict(corr=np.mean(c), age=k[0]))
    corrs = pd.DataFrame(corrs)

    fig = figure(2, 2)
    ax = sns.scatterplot(
        data=corrs, x="age", y="corr", palette=ONTOGENY_AGE_CMAP, s=5, hue="age"
    )
    legend(ax, title="Age (wks)")
    ax.set(ylim=(0, 1), ylabel="Dynamics correlation")
    sns.despine()

    saver(fig, "dynamics-correlation-scatter")
    with open(save_path / "dynamics-correlation.txt", "w") as f:
        json.dump({"correlation": corrs["corr"].median()}, f)

    save_path = save_path / "session_dynamics"
    saver = save_factory(save_path, tight_layout=True)

    for k, v in dynamics_data.items():
        tmp_df = pd.DataFrame(v)
        fig = _plot_dynamics(tmp_df, k[0], n_frames=n_frames)
        fig.suptitle(f"{k[0]} wks")
        saver(fig, f"dynamics_{k[1].parents[1].name}")
        fig = _plot_dynamics(zscore_pd(tmp_df), k[0], n_frames=n_frames)
        fig.suptitle(f"{k[0]} wks")
        saver(fig, f"dynamics_zscore_{k[1].parents[1].name}")


### ~~~ inspection of select poses ~~~ ###


def plot_poses(pre_xform_path: Path, post_xform_path: Path, save_path: Path):
    import matplotlib.pyplot as plt

    saver = save_factory(save_path, tight_layout=True)

    pre_xform = joblib.load(pre_xform_path)
    post_xform = joblib.load(post_xform_path)

    old = keyfilter(lambda k: k[0] == 103, pre_xform)
    mid = keyfilter(lambda k: k[0] == 48, pre_xform)
    young = keyfilter(lambda k: k[0] == 4, pre_xform)
    old_keys = sorted(old)
    mid_keys = sorted(mid)
    young_keys = sorted(young)

    # key-index pairs
    poses = [
        (old_keys[0], 2),
        (old_keys[1], 55),
        (old_keys[2], 0),
        (old_keys[2], 20),
        (young_keys[0], 0),
        (young_keys[0], 10),
        (young_keys[1], 17),
        (young_keys[2], 17),
        (mid_keys[0], 1),
        (mid_keys[0], 30),
        (mid_keys[1], 20),
        (mid_keys[2], 120),
        (mid_keys[3], 32),
    ]

    fig, ax = plt.subplots(len(poses), 2, figsize=(2.2, 1.2 * len(poses)))
    for i, (key, idx) in enumerate(poses):
        ax[i, 0].imshow(
            pre_xform[key][idx], interpolation="none", vmax=70, cmap="cubehelix"
        )
        ax[i, 1].imshow(
            post_xform[key][idx], interpolation="none", vmax=70, cmap="cubehelix"
        )
        ax[i, 0].set(yticks=[], xticks=[], title=f"{key[0]} wks")
        ax[i, 1].set(yticks=[], xticks=[])

    saver(fig, "selected_poses")
    return save_path


### ~~~ pre/post network mouse size predictions ~~~ ###


def compute_areas(pose_data: dict):
    def _compute_area(pose):
        return np.median(np.sum(pose > 12, axis=(1, 2)))

    return valmap(_compute_area, pose_data)


def mouse_size_predictions(
    pre_xform_path: Path, post_xform_path: Path, save_path: Path
):
    from aging.plotting import ONTOGENY_AGE_CMAP

    format_plots()
    saver = save_factory(save_path, tight_layout=True)

    pre_xform = compute_areas(joblib.load(pre_xform_path))
    post_xform = compute_areas(joblib.load(post_xform_path))

    x = pd.Series(pre_xform, name="Original")
    y = pd.Series(post_xform, name="Transformed")

    # compute correlation using pandas
    corr = x.corr(y)

    with open(save_path / "correlation.txt", "w") as f:
        json.dump({"correlation": corr}, f)

    # plot
    df = pd.DataFrame(dict(x=x, y=y))
    df["age"] = df.index.get_level_values(0)
    fig = figure(2.5, 2)
    ax = sns.scatterplot(
        data=df, x="x", y="y", palette=ONTOGENY_AGE_CMAP, s=2, hue="age"
    )
    ax = sns.regplot(data=df, x="x", y="y", color="k", scatter=False)
    ax.set(xlabel="Original area", ylabel="Transformed area")
    legend(ax, title="Age (wks)")
    sns.despine()
    saver(fig, "size_correlation")
    return save_path
