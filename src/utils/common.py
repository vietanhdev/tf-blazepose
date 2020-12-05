import numpy as np
import cv2
import random

def calculate_bbox_from_keypoints(kps, padding=0.2):
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


def random_occlusion(image, keypoints, visibility=None, rect_ratio=None, rect_color="random"):
    """Generate random rectangle to occlude points
        From BlazePose paper: "To support the prediction of invisible points, we simulate occlusions (random
        rectangles filled with various colors) during training and introduce a per-point
        visibility classifier that indicates whether a particular point is occluded and
        if the position prediction is deemed inaccurate."

    Args:
        image: Input image
        keypoints: Keypoints in format [[x1, y1], [x2, y2], ...]
        visibility [list]: List of visibilities of keypoints. 0: occluded by rectangle, 1: visible
        rect_ratio: Rect ratio wrt image width and height. Format ((min_width, max_width), (min_height, max_height))
                    Example: ((0.2, 0.5), (0.2, 0.5))
        rect_color: Scalar indicating color to fill in the rectangle

    Return:
        image: Generated image
        visibility [list]: List of visibilities of keypoints. 0: occluded by rectangle, 1: visible
    """

    if rect_ratio is None:
        rect_ratio = ((0.2, 0.5), (0.2, 0.5))

    im_height, im_width = image.shape[:2]
    print(image.shape[:2])
    rect_width = int(im_width * random.uniform(*rect_ratio[0]))
    rect_height = int(im_height * random.uniform(*rect_ratio[1]))
    print(rect_width, rect_height)
    rect_x = random.randint(0, im_width - rect_width)
    rect_y = random.randint(0, im_height - rect_height)

    gen_image = image.copy()
    if rect_color == "random":
        rect_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    gen_image = cv2.rectangle(gen_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rect_color, -1)

    if visibility is None:
        visibility = [1] * len(keypoints)
    for i in range(len(visibility)):
        if rect_x < keypoints[i][0] and keypoints[i][0] < rect_x + rect_width \
            and rect_y < keypoints[i][1] and keypoints[i][1] < rect_y + rect_height:
            visibility[i] = 0

    return gen_image, visibility


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

    # Update keypoints
    keypoints = np.array(keypoints)
    keypoints[:, 0] = keypoints[:, 0] - bbox[0][0] + pad_left
    keypoints[:, 1] = keypoints[:, 1] - bbox[0][1] + pad_top

    return cropped_image, keypoints

