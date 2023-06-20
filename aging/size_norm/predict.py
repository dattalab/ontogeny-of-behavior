import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from aging.size_norm.data import Session, unnormalize
from aging.size_norm.lightning import SizeNormModel
from typing import Union


def predict(data: Session, model: Union[str, Path, SizeNormModel], batch_size=512, **tqdm_kwargs):
    dataset = DataLoader(data, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(model, SizeNormModel):
        model = SizeNormModel.load_from_checkpoint(model, map_location=device)
    model.eval()
    model.freeze()

    output = []
    with torch.no_grad():
        for batch in tqdm(dataset, **tqdm_kwargs):
            output.append(unnormalize(model(batch.to(device))).cpu().numpy().squeeze())
    output = np.concatenate(output, axis=0)
    del dataset
    del data

    return output
