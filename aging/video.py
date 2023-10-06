import cv2
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from matplotlib.pyplot import get_cmap


def write_movie(fname: str, data: np.ndarray, cmap="cubehelix", fps=30, color_resolution=256):
    fname: Path = Path(fname).with_suffix(".mp4")
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    data_min = np.nanmin(data)
    scale = np.ptp(data)
    cmap = get_cmap(cmap, color_resolution)

    writer = iio.get_writer(
        fname,
        fps=fps,
        mode="I",
        format="FFMPEG",
        codec="libx264",
        output_params=["-crf", "29", "-preset", "slower", "-pix_fmt", "yuv420p"],
    )

    for i, frame in enumerate(map(np.nan_to_num, data)):
        img = np.uint8(cmap((frame - data_min) / scale) * 255)
        img = cv2.copyMakeBorder(img, 32, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
        img = cv2.putText(img, str(i), (0, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        writer.append_data(img)
    writer.close()


def write_movie_av(fname: str, data: np.ndarray, cmap="cubehelix", fps=30, color_resolution=256):
    fname: Path = Path(fname).with_suffix(".mp4")
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    data_min = np.nanmin(data)
    scale = np.ptp(data)
    cmap = get_cmap(cmap, color_resolution)

    with iio.imopen(fname, 'w', plugin='pyav') as file:
        # set pix_fmt
        file.init_video_stream('libx264', fps=fps, pixel_format='yuv420p')

        for i, frame in enumerate(map(np.nan_to_num, data)):
            img = np.uint8(cmap((frame - data_min) / scale) * 255)
            img = cv2.copyMakeBorder(img, 32, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
            img = cv2.putText(img, str(i), (0, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            file.write_frame(img[..., :3])