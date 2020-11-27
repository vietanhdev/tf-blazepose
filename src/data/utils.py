import os
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin
from imutils import face_utils

def normalize_landmark_point(original_point, image_size):
    '''
    original_point: (x, y)
    image_size: (W, H)
    '''
    x, y = original_point
    x /= image_size[0]
    y /= image_size[1]
    return [x, y]

def unnormalize_landmark_point(normalized_point, image_size, scale=[1,1]):
    '''
    normalized_point: (x, y)
    image_size: (W, H)
    '''
    x, y = normalized_point
    x *= image_size[0]
    y *= image_size[1]
    x *= scale[0]
    y *= scale[1]
    return [x, y]

def unnormalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.multiply(np.array(landmark), np.array(image_size))
    return landmark

def normalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.divide(landmark, np.array(image_size))
    return landmark

def draw_landmark(img, landmark):
    im_width = img.shape[1]
    im_height = img.shape[0]
    img_size = (im_width, im_height)
    landmark = landmark.reshape((-1, 2))
    unnormalized_landmark = unnormalize_landmark(landmark, img_size)
    for i in range(unnormalized_landmark.shape[0]):
        img = cv2.circle(img, (int(unnormalized_landmark[i][0]), int(unnormalized_landmark[i][1])), 2, (0,255,0), 2)
    return img

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
