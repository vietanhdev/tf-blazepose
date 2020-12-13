import random

import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

from .augmentation_utils import add_vertical_reflection

seq = [None]


def load_aug():

    def sometimes(aug): return iaa.Sometimes(0.2, aug)

    seq[0] = iaa.Sequential(
        [
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-10, 10),
                shear=(-5, 5),
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode=ia.ALL
            )),
            iaa.Sometimes(0.1, iaa.MotionBlur(k=15, angle=[-45, 45])),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 5)),
                    iaa.MedianBlur(k=(3, 5)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                    0.75, 1.5)),  # sharpen images
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
            ],
                random_order=True
            )
        ],
        random_order=True
    )


def augment_img(image, landmark=None):
    if seq[0] is None:
        load_aug()

    if landmark is None:
        image_aug = seq[0](images=np.array([image]))
        return image_aug[0]
    else:

        landmark_xy = landmark[:, :2]
        image_aug, landmark_xy = seq[0](images=np.array(
            [image]), keypoints=np.array([landmark_xy]))
        image_aug = image_aug[0]
        landmark_xy = landmark_xy[0]

        # Simulate reflection
        if random.random() < 0.1:
            image_aug = add_vertical_reflection(image_aug, landmark_xy)

        landmark[:, :2] = landmark_xy
        # draw = image_aug.copy()
        # for i in range(landmark.shape[0]):
        # 	draw = cv2.circle(draw, (int(landmark[i][0]), int(landmark[i][1])), 2, (0,255,0), 2)
        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)
        return image_aug, landmark
