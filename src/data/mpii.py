import os
import numpy as np
import cv2
import random
from tensorflow.keras.utils import Sequence
import math
import random
import json

from .utils import normalize_landmark, unnormalize_landmark, transform, draw_labelmap
from .augmentation import augment_img

class DataSequence(Sequence):

    def __init__(self, image_folder, label_file, batch_size=8, input_size=(256, 256), heatmap_size=(128, 128), heatmap_sigma=4, n_points=16, shuffle=True, augment=False, random_flip=False, random_rotate=False, random_scale_on_crop=False):

        self.batch_size = batch_size
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.image_folder = image_folder
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_scale_on_crop = random_scale_on_crop
        self.augment = augment
        self.n_points = n_points

        with open(label_file, "r") as fp:
            self.anno = json.load(fp)

        if shuffle:
            random.shuffle(self.anno)

    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return math.floor(len(self.anno) / float(self.batch_size))
        

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        batch_data = self.anno[idx * self.batch_size: (1 + idx) * self.batch_size]

        batch_image = []
        batch_landmark = []
        batch_heatmap = []

        for data in batch_data:

            # Flip 50% of images
            flip = False
            if self.random_flip and random.random() < 0.5:
                flip = True

            # Load and augment data
            image, landmark, heatmap = self.load_data(self.image_folder, data)

            batch_image.append(image)
            batch_landmark.append(landmark)
            batch_heatmap.append(heatmap)

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark)
        batch_heatmap = np.array(batch_heatmap)

        batch_image = DataSequence.preprocess_images(batch_image)
        batch_landmark = self.preprocess_landmarks(batch_landmark)

        # Prevent values from going outside [0, 1]
        batch_landmark[batch_landmark < 0] = 0
        batch_landmark[batch_landmark > 1] = 1

        return batch_image, [batch_landmark, batch_heatmap]

    @staticmethod
    def preprocess_images(images):
        # Convert color to RGB
        for i in range(images.shape[0]):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
        images = np.array(images, dtype=np.float32)
        images = (images - mean) / 255
        return images

    def preprocess_landmarks(self, landmarks):
        landmarks = landmarks.reshape((-1, 3))
        landmarks = normalize_landmark(landmarks, self.input_size)
        landmarks = landmarks.reshape((self.batch_size, -1))
        return landmarks

    def load_data(self, img_folder, data):

        # Load image
        path = os.path.join(img_folder, data["img_paths"])
        image = cv2.imread(path)

        # get center
        center = np.array(data['objpos'])
        joints = np.array(data['joint_self'])
        scale = data['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        # Flip
        if self.random_flip and random.choice([0, 1]):
            image, joints, center = self.flip(image, joints, center)

        # Scale
        if self.random_scale_on_crop:
            scale = scale * np.random.uniform(0.9, 1.1)

        # Rotate image
        if self.random_rotate and random.choice([0, 1]):
            rot = np.random.randint(-1 * 30, 30)
        else:
            rot = 0
        rot = 0 # We are having a bug in rotation

        cropimg = self.crop(image, center, scale, self.input_size, rot)
        cropimg = cv2.resize(cropimg, (self.input_size)).astype(np.uint8)

        # transform keypoints
        landmark = self.transform_kp(joints, center, scale, self.input_size, rot)
        
        # Augment image using imgaug
        if self.augment:
            landmark_xy = landmark[:, :2].astype(int)
            cropimg, landmark_xy = augment_img(cropimg, landmark_xy)
            landmark[:, :2] = landmark_xy

        # Generate heatmap
        gtmap_kps = landmark.copy()
        gtmap_kps[:, :2] = (np.array(gtmap_kps[:, :2]).astype(float)
                    * np.array(self.heatmap_size) / np.array(self.input_size)).astype(int)
        gtmap = self.generate_gtmap(gtmap_kps, self.heatmap_sigma, self.heatmap_size)

        # Uncomment following lines to debug augmentation
        # draw = cropimg.copy()
        # for i in range(len(landmark)):
        #     x = int(landmark[i][0])
        #     y = int(landmark[i][1])

        #     draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.circle(draw, (int(x), int(y)), 1, (0,0,255))

        # cv2.imshow("draw", draw)
        # cv2.imshow("gtmap", gtmap.sum(axis=2))
        # cv2.waitKey(0)

        return cropimg, landmark, gtmap

    @classmethod
    def get_kp_keys(cls):
        keys = ['r_ankle', 'r_knee', 'r_hip',
                'l_hip', 'l_knee', 'l_ankle',
                'plevis', 'thorax', 'upper_neck', 'head_top',
                'r_wrist', 'r_elbow', 'r_shoulder',
                'l_shoulder', 'l_elbow', 'l_wrist']
        return keys

    def generate_gtmap(self, joints, sigma, outres):
        npart = joints.shape[0]
        gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
        for i in range(npart):
            visibility = joints[i, 2]
            if visibility > 0:
                gtmap[:, :, i] = draw_labelmap(gtmap[:, :, i], joints[i, :], sigma)
        return gtmap

    def transform_kp(self, joints, center, scale, res, rot):
        newjoints = np.copy(joints)
        for i in range(joints.shape[0]):
            if joints[i, 0] > 0 and joints[i, 1] > 0:
                _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, res=res, invert=0, rot=rot)
                newjoints[i, 0:2] = _x
        return newjoints

    def crop(self, img, center, scale, res, rot=0):
        # Preprocessing for efficient cropping
        ht, wd = img.shape[0], img.shape[1]
        sf = scale * 200.0 / res[0]
        if sf < 2:
            sf = 1
        else:
            new_size = int(np.math.floor(max(ht, wd) / sf))
            new_ht = int(np.math.floor(ht / sf))
            new_wd = int(np.math.floor(wd / sf))
            img = cv2.resize(img, (new_wd, new_ht))
            center = center * 1.0 / sf
            scale = scale / sf

        # Upper left point
        ul = np.array(transform([0, 0], center, scale, res, invert=1))
        # Bottom right point
        br = np.array(transform(res, center, scale, res, invert=1))

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        # if not rot == 0:
        #     # Remove padding
        #     new_img = scipy.misc.imrotate(new_img, rot)
        #     new_img = new_img[pad:-pad, pad:-pad]

        new_img = cv2.resize(new_img, (res[1], res[0]))
        return new_img


    def flip(self, image, joints, center):

        joints = np.copy(joints)

        matchedParts = (
            [0, 5],  # ankle
            [1, 4],  # knee
            [2, 3],  # hip
            [10, 15],  # wrist
            [11, 14],  # elbow
            [12, 13]  # shoulder
        )

        org_height, org_width, channels = image.shape

        # flip image
        flipimage = cv2.flip(image, flipCode=1)

        # flip each joints
        joints[:, 0] = org_width - joints[:, 0]

        for i, j in matchedParts:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

        # center
        flip_center = center
        flip_center[0] = org_width - center[0]

        return flipimage, joints, flip_center
