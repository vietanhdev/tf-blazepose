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
from .augmentation import augment_img
from .augmentation_utils import random_occlusion


class DataSequence(Sequence):

    def __init__(self, image_folder, label_file, batch_size=8, input_size=(256, 256), output_heatmap=True, heatmap_size=(128, 128), heatmap_sigma=4, n_points=16, shuffle=True, augment=False, random_flip=False, random_rotate=False, random_scale_on_crop=False, clip_landmark=False, symmetry_point_ids=None):

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_heatmap = output_heatmap
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.image_folder = image_folder
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_scale_on_crop = random_scale_on_crop
        self.augment = augment
        self.n_points = n_points
        self.symmetry_point_ids = symmetry_point_ids
        self.clip_landmark = clip_landmark # Clip value of landmark to range [0, 1]

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

        batch_data = self.anno[idx *
                               self.batch_size: (1 + idx) * self.batch_size]

        batch_image = []
        batch_landmark = []
        batch_heatmap = []
        batch_pushup = []

        for data in batch_data:

            # Load and augment data
            image, landmark, heatmap, is_pushup = self.load_data(self.image_folder, data)
            
            batch_image.append(image)
            batch_landmark.append(landmark)
            batch_pushup.append(is_pushup)
            if self.output_heatmap:
                batch_heatmap.append(heatmap)

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark)
        if self.output_heatmap:
            batch_heatmap = np.array(batch_heatmap)

        batch_image = DataSequence.preprocess_images(batch_image)
        batch_landmark = self.preprocess_landmarks(batch_landmark)

        # print(batch_pushup)
        batch_pushup = np.array(batch_pushup)

        # Prevent values from going outside [0, 1]
        # Only applied for sigmoid output
        if self.clip_landmark:
            batch_landmark[batch_landmark < 0] = 0
            batch_landmark[batch_landmark > 1] = 1

        # if self.output_heatmap:
        #     return batch_image, [batch_landmark, batch_heatmap]
        # else:
        #     return batch_image, batch_landmark
        return batch_image, [batch_heatmap, batch_pushup]

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

    def preprocess_landmarks(self, landmarks):

        first_dim = landmarks.shape[0]
        landmarks = landmarks.reshape((-1, 3))
        landmarks = normalize_landmark(landmarks, self.input_size)
        landmarks = landmarks.reshape((first_dim, -1))
        return landmarks

    def load_data(self, img_folder, data):

        # Load image
        path = os.path.join(img_folder, data["image"])
        image = cv2.imread(path)

        # Load landmark and apply square cropping for image
        landmark = data["points"]
        bbox = data["bbox"]
        landmark = np.array(landmark)

        # Convert all (-1, -1) to (0, 0)
        for i in range(landmark.shape[0]):
            if landmark[i][0] == -1 and landmark[i][1] == -1:
                landmark[i, :] = [0, 0]

        # Generate visibility mask
        # visible = inside image + not occluded by simulated rectangle
        # (see BlazePose paper for more detail)
        visibility = np.ones((landmark.shape[0], 1), dtype=int)
        for i in range(len(visibility)):
            if 0 > landmark[i][0] or landmark[i][0] >= image.shape[1] \
                    or 0 > landmark[i][1] or landmark[i][1] >= image.shape[0] \
                    or (landmark[i][0] == 0 and landmark[i][1] == 0):
                visibility[i] = 0

        image, landmark = square_crop_with_keypoints(
            image, bbox, landmark, pad_value="random")
        landmark = np.array(landmark)

        # Resize image
        old_img_size = np.array([image.shape[1], image.shape[0]])
        image = cv2.resize(image, self.input_size)
        landmark = (
            landmark * np.divide(np.array(self.input_size).astype(float), old_img_size)).astype(int)

        # Horizontal flip
        # and update the order of landmark points
        if self.random_flip and random.choice([0, 1]):
            image = cv2.flip(image, 1)

            # Mark missing keypoints
            missing_idxs = []
            for i in range(landmark.shape[0]):
                if landmark[i, 0] == 0 and landmark[i, 1] == 0:
                    missing_idxs.append(i)

            # Flip landmark
            landmark[:, 0] = self.input_size[0] - landmark[:, 0]

            # Restore missing keypoints
            for i in missing_idxs:
                landmark[i, 0] = 0
                landmark[i, 1] = 0

            # Change the indices of landmark points and visibility
            if self.symmetry_point_ids is not None:
                for p1, p2 in self.symmetry_point_ids:
                    l = landmark[p1, :].copy()
                    landmark[p1, :] = landmark[p2, :].copy()
                    landmark[p2, :] = l

        # Random occlusion
        # (see BlazePose paper for more detail)
        if self.augment and random.random() < 0.2:
            landmark = landmark.reshape(-1, 2)
            image, visibility = random_occlusion(image, landmark, visibility=visibility,
                                                 rect_ratio=((0.2, 0.5), (0.2, 0.5)), rect_color="random")

        # Concatenate visibility into landmark
        visibility = np.array(visibility)
        visibility = visibility.reshape((landmark.shape[0], 1))
        landmark = np.hstack((landmark, visibility))

        if self.augment:

            # Mark missing keypoints
            missing_idxs = []
            for i in range(landmark.shape[0]):
                if landmark[i, 0] == 0 and landmark[i, 1] == 0:
                    missing_idxs.append(i)

            image, landmark = augment_img(image, landmark)

            # Restore missing keypoints
            for i in missing_idxs:
                landmark[i, 0] = 0
                landmark[i, 1] = 0


        # Generate heatmap
        gtmap = None
        if self.output_heatmap:
            gtmap_kps = landmark.copy()
            gtmap_kps[:, :2] = (np.array(gtmap_kps[:, :2]).astype(float)
                                * np.array(self.heatmap_size) / np.array(self.input_size)).astype(int)
            gtmap = gen_gt_heatmap(
                gtmap_kps, self.heatmap_sigma, self.heatmap_size)

        # # Uncomment following lines to debug augmentation
        # draw = visualize_keypoints(image, landmark, visibility, text_color=(0,0,255))
        # cv2.namedWindow("draw", cv2.WINDOW_NORMAL)
        # cv2.imshow("draw", draw)
        # if self.output_heatmap:
        #     cv2.imshow("gtmap", gtmap.sum(axis=2))
        # cv2.waitKey(0)

        landmark = np.array(landmark)
        return image, landmark, gtmap, int(data["is_pushing_up"])
