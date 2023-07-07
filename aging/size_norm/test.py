import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from aging.size_norm.data import Session, clean
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet


def test(validation_path: str | Path, model, save_folder: Path | str, max_frames=9_500) -> pd.DataFrame:
    from aging.size_norm.lightning import predict
    np.random.seed(0)
    save_folder.mkdir(exist_ok=True, parents=True)
    model.eval()
    model.freeze()

    raw_pipeline = make_pipeline(
        PCA(n_components=10),
        StandardScaler(),
    )

    pred_pipeline = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.9),
    )

    results = []

    with h5py.File(validation_path, 'r') as h5f:
        for key in h5f['data']:
            data = np.array([clean(frame) for frame in h5f['data'][key][:max_frames]])
            preds = predict(Session(data), model, batch_size=512, disable=True)

            pcs = raw_pipeline.fit_transform(data.reshape(len(data), -1))

            pred_pcs = PCA(n_components=50).fit_transform(preds.reshape(len(preds), -1))

            lr = LinearRegression()
            lr.fit(pred_pcs, pcs)
            score = lr.score(pred_pcs, pcs)

            p = lr.predict(pred_pcs)
            corr = np.diag(np.corrcoef(pcs.T, p.T)[:10, 10:]).mean()
            mse = mean_squared_error(pcs, p)

            heldout = cross_val_score(pred_pipeline, pred_pcs, pcs, cv=4, scoring='neg_mean_squared_error')
            heldout2 = cross_val_score(pred_pipeline, pred_pcs, pcs, cv=4, scoring='r2')

            results.append({
                'animal': key,
                'fit_score': score,
                'fit_mse': mse,
                'heldout_mse': -heldout.mean(),
                'heldout_score': heldout2.mean(),
                'corr': corr,
                'n_frames': max_frames,
            })

    # save dataframe
    df = pd.DataFrame(results)
    df.to_parquet(save_folder / 'behavior_validation_results.parquet')
    return df


