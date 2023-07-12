import cv2
import h5py
import torch
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
from kornia import augmentation
from typing import Union


@dataclass
class AugmentationParams:
    angle_range: Tuple[int, int] = (-80, 80)
    scale_x: Tuple[float, float] = (0.5, 1.85)
    scale_y: Tuple[float, float] = (0.5, 1.85)
    scale_z: Tuple[float, float] = (0.8, 1.2)
    translate_x: Tuple[float, float] = (-0.2, 0.2)
    translate_y: Tuple[float, float] = (-0.3, 0.3)
    height: Tuple[int, int] = (-5, 5)
    white_noise_scale: float = 3  # mm

    elastic_alpha: float = 2.0
    elastic_sigma: float = 9.0
    elastic_ksize: int = 61
    elastic_flag: bool = True

    # transform probabilities
    rotation_prob: float = 0.7
    translation_prob: float = 0.7
    scale_prob: float = 0.7
    height_offset_prob: float = 0.5
    height_scale_prob: float = 0.7
    preserve_aspect_prob: float = 0.5
    white_noise_prob: float = 0.3
    flip_prob: float = 0.5
    elastic_warp_prob: float = 0.2
    random_elastic_prob: float = 0.2
    tps_warp_prob: float = 0.2
    wall_reflection_prob: float = 0.1

    # random
    rng: random.Random = random.Random(0)


@dataclass
class TrainingPaths:
    training: str | Path
    wall_noise: str | Path
    validation: str | Path


# add noise
def white_noise(img: torch.Tensor, params: AugmentationParams):
    img = img + torch.randn_like(img, device=img.device) * params.white_noise_scale
    return torch.clamp(img, 0, None)

def normalize(img):
    return img / 100

def unnormalize(img):
    return img * 100

# specific non-linear warping
def warp_tps(img):
    pass

def warp_elastic(img):
    pass

# add wall reflections
def add_reflection(img, wall):
    wall = wall * (img < 0.2)
    return img + wall


def clean(frame, tail_ksize=11, height_thresh=12, dilate=True):
    tailfilter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail_ksize, ) * 2)
    mask = cv2.morphologyEx((frame > height_thresh).astype('uint8'), cv2.MORPH_OPEN, tailfilter)
    if dilate:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, ) * 2))

    return frame * mask

# --- augmentation pipeline ---
class Augmenter:
    def __init__(self, params: AugmentationParams, wall_noise_path):
        self.params = params
        self.rotate = augmentation.RandomRotation(params.angle_range, p=params.rotation_prob)
        self.flip = augmentation.RandomRotation([179, 181], p=params.flip_prob)
        self.translate = augmentation.RandomAffine(degrees=0, translate=(params.translate_x[1], params.translate_y[1]), p=params.translation_prob)
        self.scale = augmentation.RandomAffine(degrees=0, scale=params.scale_x + params.scale_y, p=params.scale_prob)
        self.iso_scale = augmentation.RandomAffine(degrees=0, scale=(min(params.scale_x + params.scale_y), max(params.scale_x + params.scale_y)), p=params.scale_prob)

        with h5py.File(wall_noise_path, 'r') as f:
            self.wall_noise = torch.tensor(f['wall_frames'][()], dtype=torch.float32)

        self.elastic = augmentation.RandomElasticTransform(kernel_size=(params.elastic_ksize, ) * 2, alpha=(params.elastic_alpha, ) * 2, sigma=(params.elastic_sigma, ) * 2, p=params.random_elastic_prob)

    def __call__(self, data):

        ### affine transforms ###
        if self.params.rng.random() < self.params.preserve_aspect_prob:
            data = self.iso_scale(data)
        else:
            data = self.scale(data)

        data = self.rotate(data)
        data = self.translate(data)

        height_scale = torch.rand(len(data), *(1,) * (data.ndim - 1), device=data.device) * (self.params.scale_z[1] - self.params.scale_z[0]) + self.params.scale_z[0]
        if self.params.rng.random() < self.params.height_scale_prob:
            data = data * height_scale

        offset = torch.rand(len(data), *(1,) * (data.ndim - 1), device=data.device) * (self.params.height[1] - self.params.height[0]) + self.params.height[0]
        offset = (data > 0) * offset
        if self.params.rng.random() < self.params.height_offset_prob:
            data = data + offset
            data = torch.clamp(data, 0, None)

        data = self.flip(data)

        ### non-linear transforms ###
        if self.params.elastic_flag:
            data = self.elastic(data)

        # add wall noise
        if self.params.rng.random() < self.params.wall_reflection_prob:
            inds = self.params.rng.choices(range(len(self.wall_noise)), k=len(data))
            data = add_reflection(data, self.wall_noise[inds].to(data.device).unsqueeze(1))

        # add white noise
        if self.params.rng.random() < self.params.white_noise_prob:
            data = white_noise(data, self.params)

        return normalize(data)


# --- dataset pipeline --- #
class SizeNormDataset(Dataset):
    def __init__(self, paths: TrainingPaths, aug_params: AugmentationParams):
        self.aug_params = aug_params

        with h5py.File(paths.training, 'r') as h5f:
            self.frames = h5f['training_frames'][()]

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
            with h5py.File(data, 'r') as h5f:
                self.frames = h5f['frames'][()]
        elif isinstance(data, np.ndarray):
            self.frames = data

    def __getitem__(self, idx):
        return normalize(torch.tensor(self.frames[idx], dtype=torch.float32).unsqueeze(0))

    def __len__(self):
        return len(self.frames)
