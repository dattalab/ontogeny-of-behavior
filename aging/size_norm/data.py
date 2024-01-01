import re
import cv2
import h5py
import torch
import random
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
from kornia import augmentation, morphology
from toolz import valfilter
from functools import cache
from scipy.stats import multivariate_t
from sklearn.preprocessing import LabelEncoder
from kornia.geometry.transform import get_tps_transform, warp_image_tps, resize, scale


@dataclass
class AugmentationParams:
    angle_range: Tuple[int, int] = (-80, 80)
    scale_x: Tuple[float, float] = (0.5, 1.85)
    scale_y: Tuple[float, float] = (0.5, 1.85)
    scale_z: Tuple[float, float] = (0.8, 1.2)
    shear_angles: Tuple[int, int, int, int] = (-15, 15, -15, 15)
    translate_x: Tuple[float, float] = (-0.2, 0.2)
    translate_y: Tuple[float, float] = (-0.3, 0.3)
    height: Tuple[int, int] = (-5, 5)
    white_noise_scale: float = 5  # mm
    min_threshold_range: Tuple[int, int] = (1, 20)
    max_threshold_range: Tuple[int, int] = (40, 120)

    # morphological parameters
    clean_prob: float = 0.2
    tail_ksize_range: Tuple[int, int] = (3, 15)
    dilation_ksize_range: Tuple[int, int] = (3, 11)
    morph_height_thresh_range: Tuple[int, int] = (5, 25)
    dilation_prob: float = 0.5

    elastic_alpha: float = 2.0
    elastic_sigma: float = 9.0
    elastic_ksize: int = 61
    elastic_flag: bool = True

    # transform probabilities
    rotation_prob: float = 0.7
    translation_prob: float = 0.7
    scale_prob: float = 0.7
    shear_prob: float = 0.35
    threshold_prob: float = 0.05
    height_offset_prob: float = 0.5
    height_scale_prob: float = 0.7
    preserve_aspect_prob: float = 0.5
    white_noise_prob: float = 0.2
    scaled_white_noise_prob: float = 0.35
    flip_prob: float = 0.5
    zoom_unzoom_prob: float = 0.3
    random_elastic_prob: float = 0.2
    tps_warp_prob: float = 0.35
    wall_reflection_prob: float = 0.15

    # random
    rng: random.Random = random.Random(0)


@dataclass
class TrainingPaths:
    training: str | Path
    wall_noise: str | Path
    validation: str | Path
    age_validation: str | Path


# add noise
def white_noise(img: torch.Tensor, params: AugmentationParams):
    img = img + torch.randn_like(img, device=img.device) * params.white_noise_scale
    return torch.clamp(img, 0, None)


def scaled_white_noise(img: torch.Tensor, params: AugmentationParams):
    noise = torch.randn([params.rng.randint(4, img.shape[-1])] * 2, device=img.device) * params.white_noise_scale
    return torch.clamp(img + resize(noise, img.shape[-2:]), 0, None)


def normalize(img):
    return img / 110


def unnormalize(img):
    return img * 110


# add wall reflections
def add_reflection(img, wall):
    wall = wall * (img < 0.2)
    return img + wall


def clean(frame, tail_ksize=11, height_thresh=12, dilate=True, dilation_ksize=3):
    tailfilter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail_ksize,) * 2)
    mask = cv2.morphologyEx(
        (frame > height_thresh).astype("uint8"), cv2.MORPH_OPEN, tailfilter
    )
    if dilate:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_ksize,) * 2)
        )

    return frame * mask


def zoom_unzoom(frame: torch.Tensor, rng, scale_range) -> torch.Tensor:
    scale_factor = rng.uniform(*scale_range)
    scale_factor = torch.ones((len(frame), 2), device=frame.device) * scale_factor
    scaled_frame = scale(frame, scale_factor)
    unscaled_frame = scale(scaled_frame, 1 / scale_factor)
    return unscaled_frame


def pad_vector(v):
    return torch.nn.functional.pad(v, (1, 1, 1, 1), value=0)


@cache
def make_grid(grid_size=6, batch_size=1, device=torch.device("cpu")):
    grid = torch.meshgrid(
        torch.linspace(-0.5, 0.5, grid_size),
        torch.linspace(-0.5, 0.5, grid_size),
        indexing="ij",
    )
    grid = torch.stack(grid, dim=-1).reshape(1, -1, 2).repeat(batch_size, 1, 1)
    return grid.to(device)


def sample_size_changes(parameters: dict, random_state: random.RandomState):
    age_params = random_state.choice(list(parameters.values()))
    tps_sampler = multivariate_t(*age_params['tps'])
    height_sampler = multivariate_t(*age_params['height'])
    scale_sampler = multivariate_t(*age_params['scale'])

    tps = tps_sampler.rvs(random_state=random_state)
    tps = tps.reshape(2, 4, 4)
    tps = pad_vector(torch.tensor(tps, dtype=torch.float32))

    height = np.clip(height_sampler.rvs(random_state=random_state), 0, None)
    height = torch.tensor(height.reshape(-1, 8), dtype=torch.float32)

    scale = scale_sampler.rvs(random_state=random_state)
    scale = torch.tensor(scale.reshape(-1, 2), dtype=torch.float32)

    return tps, height, scale


def age_dependent_tps_xform(frames, parameters: dict, random_state: random.RandomState):
    samples = [sample_size_changes(parameters, random_state) for _ in range(len(frames))]
    tps_vector, height, scale = zip(*samples)
    tps_vector = torch.transpose(torch.stack(tps_vector), 1, 3).to(frames.device)

    grid = make_grid(tps_vector.shape[1], len(frames), device=frames.device)

    kernel, affine = get_tps_transform(grid + tps_vector.reshape(len(frames), -1, 2), grid)
    scaled_frames = scale(frames, torch.cat(scale, dim=0))
    transformed_frames = warp_image_tps(scaled_frames, grid, kernel, affine)

    height_intermediate = resize(torch.stack(height), frames.shape[-2:])

    out = transformed_frames * height_intermediate
    return out


@cache
def get_circular_kernel(size):
    k = torch.zeros((size, size), dtype=torch.float32)
    mid = size // 2
    for i in range(size):
        for j in range(size):
            dist = (i - mid) ** 2 + (j - mid) ** 2
            if dist <= mid**2:
                k[i, j] = 1
    return k


def augmentation_clean(frames, params: AugmentationParams):
    tail_ksize = params.rng.randint(*params.tail_ksize_range)
    kernel = get_circular_kernel(tail_ksize).to(frames.device)
    dilation_ksize = params.rng.randint(*params.dilation_ksize_range)
    dilation_kernel = get_circular_kernel(dilation_ksize).to(frames.device)

    thresh = params.rng.randint(*params.morph_height_thresh_range)
    mask = (frames > thresh).float()
    mask = morphology.opening(mask, kernel, engine="convolution")

    if params.rng.random() < params.dilation_prob:
        mask = morphology.dilation(mask, dilation_kernel, engine="convolution")

    return frames * mask


# --- augmentation pipeline ---
class Augmenter:
    def __init__(self, params: AugmentationParams, wall_noise_path, tps_sampler_path):
        self.params = params
        self.rotate = augmentation.RandomRotation(
            params.angle_range, p=params.rotation_prob
        )
        self.flip = augmentation.RandomRotation([179, 181], p=params.flip_prob)
        self.translate = augmentation.RandomAffine(
            degrees=0,
            translate=(params.translate_x[1], params.translate_y[1]),
            p=params.translation_prob,
        )
        self.scale = augmentation.RandomAffine(
            degrees=0, scale=params.scale_x + params.scale_y, p=params.scale_prob
        )
        self.iso_scale = augmentation.RandomAffine(
            degrees=0,
            scale=(
                min(params.scale_x + params.scale_y),
                max(params.scale_x + params.scale_y),
            ),
            p=params.scale_prob,
        )
        self.shear = augmentation.RandomShear(params.shear_angles, p=params.shear_prob)

        with h5py.File(wall_noise_path, "r") as f:
            self.wall_noise = torch.tensor(f["wall_frames"][()], dtype=torch.float32)

        self.elastic = augmentation.RandomElasticTransform(
            kernel_size=(params.elastic_ksize,) * 2,
            alpha=(params.elastic_alpha,) * 2,
            sigma=(params.elastic_sigma,) * 2,
            p=params.random_elastic_prob,
        )

        self.tps_sampling_params = joblib.load(tps_sampler_path)

    def __call__(self, data):

        # add white noise correlated with scaling
        if self.params.rng.random() < self.params.white_noise_prob:
            data = white_noise(data, self.params)

        if self.params.rng.random() < self.params.tps_warp_prob:
            data = age_dependent_tps_xform(data, self.tps_sampling_params, self.params.rng)

        ### affine transforms ###
        if self.params.rng.random() < self.params.preserve_aspect_prob:
            data = self.iso_scale(data)
        else:
            data = self.scale(data)

        data = self.rotate(data)
        data = self.translate(data)
        data = self.shear(data)

        height_scale = (
            torch.rand(len(data), *(1,) * (data.ndim - 1), device=data.device)
            * (self.params.scale_z[1] - self.params.scale_z[0])
            + self.params.scale_z[0]
        )
        if self.params.rng.random() < self.params.height_scale_prob:
            data = data * height_scale

        data = self.flip(data)

        ### non-linear transforms ###
        if self.params.elastic_flag:
            data = self.elastic(data)

        # shrink then re-expand to simulate data loss and changes in noise scaling
        if self.params.rng.random() < self.params.zoom_unzoom_prob:
            data = zoom_unzoom(data, self.params.rng, self.params.scale_x)

        # add wall noise
        if self.params.rng.random() < self.params.wall_reflection_prob:
            inds = self.params.rng.choices(range(len(self.wall_noise)), k=len(data))
            data = add_reflection(
                data, self.wall_noise[inds].to(data.device).unsqueeze(1)
            )

        # add white noise uncorrelated with scaling
        if self.params.rng.random() < self.params.white_noise_prob:
            data = white_noise(data, self.params)

        # add scaled white noise
        if self.params.rng.random() < self.params.scaled_white_noise_prob:
            data = scaled_white_noise(data, self.params)

        # min threshold
        if self.params.rng.random() < self.params.threshold_prob:
            threshold = self.params.rng.uniform(*self.params.min_threshold_range)
            data[data < threshold] = 0
        # max threshold
        if self.params.rng.random() < self.params.threshold_prob:
            threshold = self.params.rng.uniform(*self.params.max_threshold_range)
            data = torch.clamp_max(data, threshold)

        # morphologically clean the data
        if self.params.rng.random() < self.params.clean_prob:
            data = augmentation_clean(data, self.params)

        # run height offset after thresholding
        offset = (
            torch.rand(len(data), *(1,) * (data.ndim - 1), device=data.device)
            * (self.params.height[1] - self.params.height[0])
            + self.params.height[0]
        )
        offset = (data > 0) * offset
        if self.params.rng.random() < self.params.height_offset_prob:
            data = data + offset
            data = torch.clamp_min(data, 0)

        return normalize(data)


# --- dataset pipeline --- #
class SizeNormDataset(Dataset):
    def __init__(self, paths: TrainingPaths):
        with h5py.File(paths.training, "r") as h5f:
            self.frames = h5f["training_frames"][()]

        # remove "bad" images
        cleaning_metadata = joblib.load(
            Path(paths.training).parent / "cleaning_metadata.p"
        )
        dump = valfilter(lambda x: x == "remove", cleaning_metadata)
        mask = np.ones(len(self.frames), dtype=bool)
        mask[list(dump)] = 0
        self.frames = self.frames[mask]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        _input = self.frames[idx]
        target = normalize(torch.tensor(clean(_input), dtype=torch.float32))
        _input = torch.tensor(_input, dtype=torch.float32).view(1, 1, *_input.shape)

        return _input.view(1, *_input.shape[-2:]), target.unsqueeze(0)


class Session(Dataset):
    def __init__(self, data: Union[str, Path, np.ndarray]):
        if isinstance(data, (str, Path)):
            self.path = data
            with h5py.File(data, "r") as h5f:
                self.frames = h5f["frames"][()]
        elif isinstance(data, np.ndarray):
            self.frames = data

    def __getitem__(self, idx):
        return normalize(
            torch.tensor(self.frames[idx], dtype=torch.float32).unsqueeze(0)
        )

    def __len__(self):
        return len(self.frames)


def str_to_age(s):
    nums = int(re.match(r"\d+", s).group())
    if "m" in s:
        return nums * 30 / 7
    return nums


class AgeClassifierDataset(Dataset):
    def __init__(
        self, data_path: str | Path, fake_len, seed=0, y_type="class", model=None
    ):
        if model is not None:
            from aging.size_norm.lightning import predict
        rng = np.random.RandomState(seed)
        self.y_type = y_type
        data = {}
        with h5py.File(data_path, "r") as h5f:
            for key in filter(lambda x: "22month" not in x, h5f):
                # by supplying a model, we can apply the model to the data just once
                if model is not None:
                    data[key] = predict(Session(h5f[key][()]), model, batch_size=256)
                else:
                    data[key] = h5f[key][()]
        if y_type == "class":
            y = np.concatenate([[key] * len(data[key]) for key in data])
            y = LabelEncoder().fit_transform(y)
        elif y_type == "continuous":
            y = np.log(
                np.concatenate([[str_to_age(key)] * len(data[key]) for key in data])
            )
        idx = rng.permutation(np.arange(len(y)))
        # TODO: make sure I have same number of datapoints for each class
        #    looks like this is true by default
        self.data = np.concatenate(list(data.values()), axis=0)[idx]
        self.y = y[idx]
        if fake_len is None or fake_len < 0:
            self.fake_len = len(self.data)
        else:
            self.fake_len = fake_len

    def __getitem__(self, idx):
        y_out = self.y[idx % len(self.data)]
        if self.y_type == "continuous":
            y_out = torch.tensor(y_out, dtype=torch.float32)
        return (
            normalize(
                torch.tensor(
                    self.data[idx % len(self.data)], dtype=torch.float32
                ).unsqueeze(0)
            ),
            y_out,
        )

    @property
    def n_classes(self):
        return len(np.unique(self.y))

    def __len__(self):
        return self.fake_len


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return min(map(len, self.datasets))
