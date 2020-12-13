import json
import math
import os
import random

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from ..utils.heatmap import gen_gt_heatmap
from ..utils.keypoints import normalize_landmark
from ..utils.pre_processing import square_crop_with_keypoints
from ..utils.visualizer import visualize_keypoints
from .augmentation2 import augment_img
from .augmentation_utils import random_occlusion


class DataSequence(Sequence):

    def __init__(self, image_folder, label_file, batch_size=8, input_size=(256, 256), shuffle=True, augment=False, random_flip=False, random_rotate=False, random_scale_on_crop=False):

        self.batch_size = batch_size
        self.input_size = input_size
        self.image_folder = image_folder
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_scale_on_crop = random_scale_on_crop
        self.augment = augment

        with open(label_file, "r") as fp:
            self.anno = json.load(fp)

        if shuffle:
            random.shuffle(self.anno)

    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return math.ceil(len(self.anno) / float(self.batch_size))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        batch_data = self.anno[idx * self.batch_size: (1 + idx) * self.batch_size]

        batch_image = []
        batch_y = []

        for data in batch_data:

            # Load and augment data
            image, y = self.load_data(self.image_folder, data)
            batch_image.append(image)
            batch_y.append(y)

        batch_image = np.array(batch_image)
        batch_y = np.array(batch_y)

        batch_image = DataSequence.preprocess_images(batch_image)

        return batch_image, batch_y

    @staticmethod
    def preprocess_images(images):
        # Convert color to RGB
        for i in range(images.shape[0]):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float)
        images = np.array(images, dtype=np.float32)
        images = images / 255.0
        images -= mean
        return images


    def load_data(self, img_folder, data):

        # Load image
        path = os.path.join(img_folder, data["image"])
        image = cv2.imread(path)

        # Resize image
        image = cv2.resize(image, self.input_size)

        is_pushing_up = int(data["is_pushing_up"])

        # Horizontal flip
        # and update the order of landmark points
        if self.random_flip and random.choice([0, 1]):
            image = cv2.flip(image, 1)

        if self.augment:
            image = augment_img(image, not is_pushing_up)

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)

        

        return image, is_pushing_up
