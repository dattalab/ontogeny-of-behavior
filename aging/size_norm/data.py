import cv2
import h5py
import torch
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
from kornia.geometry import transform
from torchvision.transforms import ElasticTransform
from toolz import concatv
from typing import Union


@dataclass
class AugmentationParams:
    angle_range: Tuple[int, int] = (-80, 80)
    scale_x: Tuple[float, float] = (0.5, 1.85)
    scale_y: Tuple[float, float] = (0.5, 1.85)
    translate_x: Tuple[float, float] = (-0.2, 0.2)
    translate_y: Tuple[float, float] = (-0.3, 0.3)
    height: Tuple[int, int] = (-5, 5)
    white_noise_scale: float = 3  # mm

    elastic_alpha: float = 80.0
    elastic_sigma: float = 5.0

    # transform probabilities
    affine_prob: float = 0.7  # of each component
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
    training: Union[str, Path]
    wall_noise: Union[str, Path]


# affine transforms
def affine(img, params: AugmentationParams):
    # scaling
    if params.rng.random() < params.preserve_aspect_prob:
        scale = torch.rand(1, 1) * (max(concatv(params.scale_x, params.scale_y)) - min(concatv(params.scale_x, params.scale_y))) + min(concatv(params.scale_x, params.scale_y))
    else:
        scale = torch.rand(1, 2)
        scale[..., 0] = scale[..., 0] * (params.scale_x[1] - params.scale_x[0]) + params.scale_x[0]
        scale[..., 1] = scale[..., 1] * (params.scale_y[1] - params.scale_y[0]) + params.scale_y[0]
    mode = params.rng.choice(['nearest', 'bilinear'])
    if params.rng.random() < params.affine_prob:
        img = transform.scale(img, scale, mode=mode)

    # rotation
    angle = torch.rand(1) * (params.angle_range[1] - params.angle_range[0]) + params.angle_range[0]
    mode = params.rng.choice(['nearest', 'bilinear'])
    if params.rng.random() < params.affine_prob:
        img = transform.rotate(img, angle, mode=mode)

    # translation
    translate = torch.rand(1, 2)
    translate[..., 0] = translate[..., 0] * (params.translate_x[1] - params.translate_x[0]) + params.translate_x[0]
    translate[..., 1] = translate[..., 1] * (params.translate_y[1] - params.translate_y[0]) + params.translate_y[0]
    translate = translate * torch.tensor(img.shape[-2:])
    mode = params.rng.choice(['nearest', 'bilinear'])
    if params.rng.random() < params.affine_prob:
        img = transform.translate(img, translate, mode=mode)

    # offset
    offset = torch.rand(1) * (params.height[1] - params.height[0]) + params.height[0]
    offset = (img > 0) * offset
    if params.rng.random() < params.affine_prob:
        img = img + offset
        img = torch.clamp(img, 0, None)

    return img

def flip(img):
    return transform.vflip(transform.hflip(img))

# add noise
def white_noise(img, params: AugmentationParams):
    img = img + torch.randn_like(img) * params.white_noise_scale
    return torch.clamp(img, 0, None)

def normalize(img):
    return img / 100

def unnormalize(img):
    return img * 100

# random non-linear warping
def random_elastic(img, params: AugmentationParams):
    xf = ElasticTransform(params.elastic_alpha, params.elastic_sigma)
    return xf(img)

# specific non-linear warping
def warp_tps(img):
    pass

def warp_elastic(img):
    pass

# add wall reflections
def add_reflection(img, wall):
    wall = wall * (img == 0)
    return img + wall


def clean(frame, tail_ksize=11):
    tailfilter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail_ksize, ) * 2)
    mask = cv2.morphologyEx((frame > 0).astype('uint8'), cv2.MORPH_OPEN, tailfilter)

    return frame * mask


# --- dataset and augmentation pipeline --- #
class SizeNormDataset(Dataset):
    def __init__(self, paths: TrainingPaths, aug_params: AugmentationParams):
        self.aug_params = aug_params

        with h5py.File(paths.training, 'r') as h5f:
            self.frames = h5f['training_frames'][()]

        with h5py.File(paths.wall_noise, 'r') as h5f:
            self.wall_noise = h5f['wall_frames'][()]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        _input = self.frames[idx]
        target = normalize(torch.tensor(clean(_input), dtype=torch.float32))
        _input = torch.tensor(_input, dtype=torch.float32).view(1, 1, *_input.shape)

        # affine transforms
        _input = affine(_input, self.aug_params)

        if self.aug_params.rng.random() < self.aug_params.flip_prob:
            _input = flip(_input)

        # random non-linear transforms
        if self.aug_params.rng.random() < self.aug_params.random_elastic_prob:
            _input = random_elastic(_input, self.aug_params)

        # then add the noise (wall, white, etc.)
        if self.aug_params.rng.random() < self.aug_params.white_noise_prob:
            _input = white_noise(_input, self.aug_params)

        if self.aug_params.rng.random() < self.aug_params.wall_reflection_prob:
            wall = self.aug_params.rng.choice(self.wall_noise)
            _input = add_reflection(_input, torch.tensor(wall).view(_input.shape))

        _input = normalize(_input)

        return _input.view(1, *_input.shape[-2:]), target.unsqueeze(0)


class Session(Dataset):
    def __init__(self, data: Union[str, Path, np.ndarray]):
        if isinstance(data, (str, Path)):
            self.path = data
            self.h5f = h5py.File(data, 'r')
            self.frames = self.h5f['frames'][()]
        elif isinstance(data, np.ndarray):
            self.frames = data

    def __getitem__(self, idx):
        return normalize(torch.tensor(self.frames[idx], dtype=torch.float32).unsqueeze(0))

    def __len__(self):
        return len(self.frames)

    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()