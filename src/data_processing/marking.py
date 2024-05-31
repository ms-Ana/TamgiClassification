import cv2
import numpy as np


def find_bounding_box(image_array: np.array):
    # Find where the white pixels are
    white_pixels = np.where(image_array > 128)

    # If there are no white pixels, return None
    if not white_pixels[0].size or not white_pixels[1].size:
        return None

    # Get the bounding box coordinates
    x_min, y_min = np.min(white_pixels[1]), np.min(white_pixels[0])
    x_max, y_max = np.max(white_pixels[1]), np.max(white_pixels[0])

    # Return the bounding box as (x, y, width, height)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def get_detection(mask: np.ndarray):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return find_bounding_box(mask)
