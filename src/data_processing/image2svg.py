import os
import subprocess
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def img2svg(input_path: Union[str, Path], save_path: Union[str, Path]):
    """Convert image .jpg and .png to .svg

    Args:
        input_path (Union[str, Path]): path to image file(.jpg, .png)
        save_path (Union[str, Path]): path to save output image(.svg)
    """
    image = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY)
    image = np.where(image > 128, 255, 0).astype(np.uint8)
    tmp_bmp = os.path.splitext(save_path)[0] + ".bmp"
    cv2.imwrite(tmp_bmp, image)
    subprocess.call(f"potrace {tmp_bmp} -b svg", shell=True)
    os.remove(tmp_bmp)


def dir_img2svg(image_dir: Union[str, Path], save_dir_path: Union[str, Path]):
    """Apply img2svg to directory with images

    Args:
        image_dir (Union[str, Path]): path to directory with images (.jpg, .png)
        save_dir_path (Union[str, Path]): path to directory to save output images (.svg) (could not exists)
    """
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    for img_path in tqdm(os.listdir(image_dir)):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img2svg(
            os.path.join(image_dir, img_path),
            os.path.join(save_dir_path, img_name + ".svg"),
        )


CONFIG = {
    "image_dir": "/home/ana/University/Tamgi/data/tamgi_images",
    "save_dir_path": "/home/ana/University/Tamgi/data/tamgi_image_svg",
}

if __name__ == "__main__":
    dir_img2svg(**CONFIG)
