import random
import numpy as np
import cv2


def add_vertical_reflection(image, keypoints, min_height=0.1):
    """Add vertical reflection

    Args:
        image: Input image
        keypoints: Keypoints
        min_height [int]: Min height ratio of reflection (over image height)

    Return:
        Augmented image
    """

    im_height = image.shape[0]
    max_y = np.max(np.array(keypoints)[:, 1])
    reflection_height = min(im_height - max_y - 1, max_y)

    if reflection_height < min_height * im_height:
        return image

    alpha = random.uniform(0.5, 0.9)
    beta = (1.0 - alpha)
    image[max_y:max_y+reflection_height, :, :] = cv2.addWeighted(image[max_y:max_y+reflection_height, :, :],
                                                                 alpha,
                                                                 cv2.flip(
                                                                     image[max_y-reflection_height:max_y, :, :], 0),
                                                                 beta, 0.0)

    return image


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
    rect_width = int(im_width * random.uniform(*rect_ratio[0]))
    rect_height = int(im_height * random.uniform(*rect_ratio[1]))
    rect_x = random.randint(0, im_width - rect_width)
    rect_y = random.randint(0, im_height - rect_height)

    gen_image = image.copy()
    if rect_color == "random":
        rect_color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
    gen_image = cv2.rectangle(gen_image, (rect_x, rect_y),
                              (rect_x + rect_width, rect_y + rect_height), rect_color, -1)

    if visibility is None:
        visibility = [1] * len(keypoints)
    for i in range(len(visibility)):
        if rect_x < keypoints[i][0] and keypoints[i][0] < rect_x + rect_width \
                and rect_y < keypoints[i][1] and keypoints[i][1] < rect_y + rect_height:
            visibility[i] = 0

    return gen_image, visibility
