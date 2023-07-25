import h5py
import torch
import numpy as np
from pathlib import Path
from toolz import valmap, reduce
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from aging.behavior.scalars import compute_scalars
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from aging.size_norm.data import Session, clean, AgeClassifierDataset, unnormalize


def _predict(model, data: Dataset, batch_size=512):
    yhat = []
    dataset = DataLoader(data, batch_size=batch_size, shuffle=False)
    out = defaultdict(list)
    with torch.no_grad():
        for x_batch in dataset:
            if isinstance(x_batch, (tuple, list)):
                x_batch, *y_batch = x_batch
                for i, y in enumerate(y_batch):
                    out[i].append(y.cpu().numpy().squeeze())
            yhat.append(
                unnormalize(model(x_batch.to(model.device))).cpu().numpy().squeeze()
            )
    yhat = np.concatenate(yhat)
    out = valmap(np.concatenate, out)
    return yhat, out


def dynamics_correlation(
    validation_path: str | Path, model, max_frames=9_500
) -> float: 
    model.eval()
    model.freeze()

    results = []

    # load in each session, compute length, width, height of animal
    with h5py.File(validation_path, "r") as h5f:
        for key in h5f["data"]:
            data = h5f['data'][key][:max_frames]
            preds, _ = _predict(model, Session(data), batch_size=512)
            data = np.array(list(map(clean, data)))
            data_scalars = compute_scalars(data, is_recon=False)
            preds_scalars = compute_scalars(preds, is_recon=False)
            mask = reduce(lambda x, y: np.logical_or(x, y), valmap(np.isnan, data_scalars).values())
            mask |= reduce(lambda x, y: np.logical_or(x, y), valmap(np.isnan, preds_scalars).values())

            corr = {k: np.corrcoef(v[~mask], preds_scalars[k][~mask])[0, 1] for k, v in data_scalars.items()}
            results.append(np.mean(list(corr.values())))
    return np.mean(results)


# run an age classifier on the validation set
def classify_age(validation: str | Dataset, model):
    if isinstance(validation, str):
        validation = AgeClassifierDataset(validation, None)
    model.eval()
    model.freeze()

    yhat, out = _predict(model, validation)

    pipeline = make_pipeline(
        PCA(n_components=15),
        StandardScaler(),
        LogisticRegression(max_iter=200),
    )

    scores = cross_val_score(
        pipeline,
        yhat.reshape(len(yhat), -1),
        out[0],
        cv=StratifiedKFold(6),
        scoring="accuracy",
    )
    return np.mean(scores)
