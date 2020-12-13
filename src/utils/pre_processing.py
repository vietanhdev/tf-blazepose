import numpy as np
import cv2
import random

def calculate_bbox_from_keypoints(kps, padding=0.1):
    """Estimate body bounding box from all body keypoints

    Args:
        kps: Keypoints. Shape: (n, 2)
        padding: Padding the smallest keypoint bounding box to form body bounding box
    """

    kps = np.array(kps)
    min_x = np.min(kps[:, 0])
    min_y = np.min(kps[:, 1])
    max_x = np.max(kps[:, 0])
    max_y = np.max(kps[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    x1 = min_x - padding * width
    x2 = max_x + padding * width
    y1 = min_y - padding * height
    y2 = max_y + padding * height

    return [[x1, y1], [x2, y2]]


def square_padding(im, desired_size=800, return_padding=False):

    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    if not return_padding:
        return new_im
    else:
        h, w = new_im.shape[:2]
        padding = (top / h, left / w, bottom / h, right / w)
        return new_im, padding


def square_crop_with_keypoints(image, bbox, keypoints, pad_value=0):
    """Square crop an image knowing a bounding box. This function also update keypoints accordingly
    Steps: Extend bbox to a square -> Pad image -> Crop image -> Recalculate keypoints 

    Args:
        image: Input image
        bbox: Bounding box. Shape: (2, 2), Format: [[x1, y1], [x2, y2]]
        keypoints: Keypoints in format [[x1, y1], [x2, y2], ...]
        pad_value: Scalar indicating padding color
    
    Returns:
        cropped_image, keypoints
    """

    bbox_width = bbox[1][0] - bbox[0][0]
    bbox_height = bbox[1][1] - bbox[0][1]
    im_height, im_width = image.shape[:2]

    if bbox_width > bbox_height: # Padding on y-axis
        pad = int((bbox_width - bbox_height) / 2)
        bbox[0][1] -= pad
        bbox[1][1] = bbox[0][1] + bbox_width
    elif bbox_height > bbox_width: # Padding on x-axis
        pad = int((bbox_height - bbox_width) / 2)
        bbox[0][0] -= pad
        bbox[1][0] = bbox[0][0] + bbox_height

    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if bbox[0][0] < 0:
        pad_left = -bbox[0][0]
        bbox[0][0] = 0
        bbox[1][0] += pad_left
    if bbox[0][1] < 0:
        pad_top = -bbox[0][1]
        bbox[0][1] = 0
        bbox[1][1] += pad_top
    if bbox[1][0] >= im_width:
        pad_right = bbox[1][0] - im_width + 1
    if bbox[1][1] >= im_height:
        pad_bottom = bbox[1][1] - im_height + 1
    
    if pad_value == "random":
        pad_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)

    cropped_image = padded_image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # Mark missing keypoints
    keypoints = np.array(keypoints)
    missing_idxs = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 0] == 0 and keypoints[i, 1] == 0:
            missing_idxs.append(i)

    # Update keypoints
    keypoints[:, 0] = keypoints[:, 0] - bbox[0][0] + pad_left
    keypoints[:, 1] = keypoints[:, 1] - bbox[0][1] + pad_top

    # Restore missing keypoints
    for i in missing_idxs:
        keypoints[i, 0] = 0
        keypoints[i, 1] = 0

    return cropped_image, keypoints

