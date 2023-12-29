import json
import joblib
import numpy as np
import seaborn as sns
from pathlib import Path
from shutil import copyfile
from sklearn.svm import SVC
from toolz import partial, compose, valmap
from sklearn.pipeline import make_pipeline
from aging.size_norm.util import multi_stage_pca
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from aging.plotting import format_plots, figure, save_factory
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold


### ~~~ age classification from raw or size-normalized images ~~~ ###

def transformed_file_exists(model_path: Path):
    return (model_path.parent / "age_classification" / "transformed_age_classification_poses.p.gz").exists()


def transform_data(data_path: Path, model_path: Path | str):
    import torch
    from aging.size_norm.data import Session
    from aging.size_norm.lightning import predict

    if isinstance(model_path, str):
        model_path = Path(model_path)

    save_path = model_path.parent / "age_classification"
    save_path.mkdir(exist_ok=True)
    save_file = save_path / "transformed_age_classification_poses.p.gz"

    # short circuit if already exists
    if save_file.exists():
        return save_file

    # load data
    data = joblib.load(data_path)

    model = torch.jit.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    predict_fun = compose(np.uint8, np.round, partial(predict, model=model, disable=True), Session)
    data = valmap(predict_fun, data)

    # save data inside folder of model_path
    joblib.dump(data, save_file, compress=3)
    return save_file


# perform age classification
def age_classifiers(data_path: Path):
    """data_path must point to SNN transformed data"""
    data = joblib.load(data_path)

    # has the same dict structure as data
    pcs = multi_stage_pca(data)
    X = np.concatenate(list(pcs.values()), axis=0)
    Y = np.concatenate(list(map(lambda x: np.repeat(x[0][0], len(x[1])), data.items())))

    gridsearch_partial = partial(
        GridSearchCV,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
    )
    crossval_partial = partial(
        cross_val_predict,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1,
    )

    # run SVC gridsearch
    pipeline = make_pipeline(StandardScaler(), SVC())
    grid = gridsearch_partial(
        pipeline,
        param_grid={
            "svc__C": [0.1, 0.5, 1, 5, 10],
        },
    )
    grid.fit(X, Y)

    predictions = crossval_partial(grid.best_estimator_, X, Y)
    svc_cm = confusion_matrix(Y, predictions, normalize="pred")
    svc_acc = accuracy_score(Y, predictions)

    # run RF gridsearch
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    grid = gridsearch_partial(
        pipeline,
        param_grid={
            "randomforestclassifier__n_estimators": [10, 50, 100, 200],
        },
    )
    grid.fit(X, Y)

    predictions = crossval_partial(grid.best_estimator_, X, Y)
    rf_cm = confusion_matrix(Y, predictions, normalize="pred")
    rf_acc = accuracy_score(Y, predictions)

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
def plot_classification(classifier_results: dict, save_path: Path):
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
    )
    ax.set(
        xlabel="Predicted Age (wks)",
        ylabel="True Age (wks)",
        aspect="equal",
        title="SVC Age Classification",
    )
    saver(fig, "svc_confusion_matrix")

    # plot rf confusion matrix
    fig = figure(2.4, 2)
    ax = sns.heatmap(
        classifier_results["rf"]["cm"],
        annot=False,
        cmap="cubehelix",
        cbar_kws={"label": "Proportion"},
        yticklabels=False,
        xticklabels=False,
    )
    ax.set(
        xlabel="Predicted Age (wks)",
        ylabel="True Age (wks)",
        aspect="equal",
        title="RF Age Classification",
    )
    saver(fig, "rf_confusion_matrix")

    # write accuracy to json file
    with open(save_path / "accuracy.txt", "w") as f:
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
def link_results_folder(saved_path: Path, model_path: Path):
    # link saved_path to a folder inside model_path
    link_path = model_path.parent / saved_path.name
    if link_path.exists():
        link_path.unlink()
    link_path.symlink_to(saved_path)

    # try to copy the config file over to the saved_path
    config_path = model_path.parent / "config.toml"
    if config_path.exists():
        copyfile(config_path, saved_path / "config.toml")

