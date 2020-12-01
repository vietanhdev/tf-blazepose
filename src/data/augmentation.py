import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

seq = [None]


def load_aug():

    def sometimes(aug): return iaa.Sometimes(0.2, aug)

    seq[0] = iaa.Sequential(
        [
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                iaa.CropAndPad(
                    percent=(-0.1, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                ),
                iaa.Crop(
                    percent=0.1,
                    keep_size=True
                ),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 0.1)),
                    iaa.AverageBlur(k=(2, 3)),
                    iaa.MedianBlur(k=(1, 3)),
                    iaa.Sharpen(alpha=(0, 0.2), lightness=(
                                0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 0.2), strength=(
                        0, 0.25)),  # emboss images
                    # search either for all edges or for directed edges,
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(
                        0.0, 0.01*255), per_channel=0.5),
                    # change hue and saturation
                    iaa.AddToHueAndSaturation((-10, 10)),
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                ]),
                # improve or worsen the contrast
                iaa.contrast.LinearContrast(
                    (0.5, 2.0), per_channel=0.1),
                iaa.Grayscale(alpha=(0.0, 0.1)),
                iaa.AdditiveLaplaceNoise(scale=0.01*255),
                iaa.AdditivePoissonNoise(lam=2),
                iaa.Multiply(mul=(0.9, 1.1)),
                iaa.Dropout(p=(0.1, 0.2)),
                iaa.CoarseDropout(p=0.1, size_percent=0.05),
                iaa.LinearContrast(),
                iaa.AveragePooling(2),
                iaa.MotionBlur(k=3),
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
        image_aug, landmark = seq[0](images=np.array(
            [image]), keypoints=np.array([landmark]))
        image_aug = image_aug[0]
        landmark = landmark[0]

        # draw = image_aug.copy()
        # for i in range(landmark.shape[0]):
        # 	draw = cv2.circle(draw, (int(landmark[i][0]), int(landmark[i][1])), 2, (0,255,0), 2)
        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)
        return image_aug, landmark
