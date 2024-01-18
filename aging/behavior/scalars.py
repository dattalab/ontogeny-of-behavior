import cv2
import numpy as np
from copy import deepcopy
from toolz import valmap


def insert_nans(timestamps, data, fps=30):
    df_timestamps = np.diff(np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))

    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)

    if filled_data.ndim == 1:
        isvec = True
        filled_data = filled_data[:, None]
    else:
        isvec = False
    _, nfeatures = filled_data.shape

    for idx in fill_idx[::-1]:
        if idx < len(missing_frames):
            ninserts = int(missing_frames[idx] - 1)
            data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
            insert_timestamps = timestamps[idx - 1] + \
                np.cumsum(np.ones(ninserts,) * 1.0 / fps)
            filled_data = np.insert(filled_data, idx,
                                    np.ones((ninserts, nfeatures)) * np.nan, axis=0)
            filled_timestamps = np.insert(
                filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def im_moment_features(frame, height_thresh=10):
    frame_mask = frame > height_thresh
    cnts, _ = cv2.findContours(frame_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = np.array([cv2.contourArea(x) for x in cnts])
    if tmp.size == 0:
        return None
    mouse_cnt = cnts[tmp.argmax()]
    tmp = cv2.moments(mouse_cnt)
    num = 2*tmp['mu11']
    den = tmp['mu20']-tmp['mu02']

    common = np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    if tmp['m00'] == 0:
        features = {
            'orientation': np.nan,
            'centroid': np.nan,
            'axis_length': [np.nan, np.nan]}
    else:
        features = {
            'orientation': -.5*np.arctan2(num, den),
            'centroid': [tmp['m10']/tmp['m00'], tmp['m01']/tmp['m00']],
            'axis_length': [2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                            2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
        }

    return features


def pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def compute_scalars(frames, centroid=None, true_depth=None, is_recon=True, height_thresh=10, clean_flag=False):
    from aging.size_norm.data import clean

    convert_mm = not (centroid is None or true_depth is None)
    if convert_mm:
        centroid_mm = pxs_to_mm(centroid, true_depth=true_depth)
        centroid_mm_shift = pxs_to_mm(centroid + 1, true_depth=true_depth)
        px_to_mm = np.abs(centroid_mm_shift - centroid_mm)

    width = []
    length = []
    height = []
    area = []
    for i, frame in enumerate(frames):
        if clean_flag:
            frame = clean(frame, height_thresh=height_thresh)
        # compute ellipse
        feats = im_moment_features(frame, height_thresh)
        if feats is None:
            width.append(np.nan)
            length.append(np.nan)
            height.append(np.nan)
        else:
            w = np.min(feats['axis_length'])
            l = np.max(feats['axis_length'])

            if convert_mm:
                w = w * px_to_mm[i, 1]
                l = l * px_to_mm[i, 0]

            width.append(w)
            length.append(l)
            height.append(np.mean(frame[(frame > height_thresh) & (frame < 110)]))
        _area = np.sum((frame > height_thresh) & (frame < 110), dtype='float32')
        if convert_mm:
            _area *= px_to_mm[i].mean()
        area.append(_area)
    pre_key = 'recon_' if is_recon else ''
    out = {
        pre_key + 'width': width,
        pre_key + 'length': length,
        pre_key + 'height': height,
        pre_key + 'area': area,
    }
    return valmap(np.array, out)