import os
import random
import numpy as np
import cv2


def unnormalize_landmark(landmark, image_size):
    """Unnormalize landmark by image size

    Args:
        landmark: Normalized keypoints in format [[x1, y1], [x2, y2], ...]
        image_size: Image size in format (width, height)

    Returns:
        Unnormalized landmark
    """
    image_size = np.array(image_size)
    landmark[:, :2] = np.multiply(
        np.array(landmark[:, :2]), np.array(image_size).reshape((1, 2)))
    return landmark


def normalize_landmark(landmark, image_size):
    """Normalize landmark by image size

    Args:
        landmark: Keypoints in format [[x1, y1], [x2, y2], ...]
        image_size: Image size in format (width, height)

    Returns:
        Normalized landmark
    """
    image_size = np.array(image_size)
    landmark = np.array(landmark)
    landmark = landmark.astype(float)
    landmark[:, :2] = np.divide(
        landmark[:, :2], np.array(image_size).reshape((1, 2)))
    return landmark


