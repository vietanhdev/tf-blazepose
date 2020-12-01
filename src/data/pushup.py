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

    def __init__(self, image_folder, label_file, batch_size=8, input_size=(256, 256), output_heatmap=True, heatmap_size=(128, 128), heatmap_sigma=4, n_points=16, shuffle=True, augment=False, random_flip=False, random_rotate=False, random_scale_on_crop=False):

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

        with open(label_file, "r") as fp:
            anno = json.load(fp)["labels"]

            # Only keep data which contains human
            self.anno = [d for d in anno if d.get("contains_person", True)]

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

            # Load and augment data
            image, landmark, heatmap = self.load_data(self.image_folder, data)

            batch_image.append(image)
            batch_landmark.append(landmark)
            if self.output_heatmap:
                batch_heatmap.append(heatmap)

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark)
        if self.output_heatmap:
            batch_heatmap = np.array(batch_heatmap)

        batch_image = DataSequence.preprocess_images(batch_image)
        batch_landmark = self.preprocess_landmarks(batch_landmark)

        # Prevent values from going outside [0, 1]
        batch_landmark[batch_landmark < 0] = 0
        batch_landmark[batch_landmark > 1] = 1

        if self.output_heatmap:
            return batch_image, [batch_landmark, batch_heatmap]
        else:
            return batch_image, batch_landmark

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

        # Add visibility output
        # TODO: Support visibility in the dataset in the future
        landmarks = landmarks.reshape(-1, 2)
        visibility = np.zeros((landmarks.shape[0], 1))
        landmarks = np.hstack((landmarks, visibility))

        landmarks = landmarks.reshape((-1, 3))
        landmarks = normalize_landmark(landmarks, self.input_size)
        landmarks = landmarks.reshape((self.batch_size, -1))
        return landmarks

    def load_data(self, img_folder, data):

        # Load image
        path = os.path.join(img_folder, data["image"])
        image = cv2.imread(path)

        # Normalize landmark for ease of processing
        # After this step, landmark points will have
        # x and y in range of [0, 1]
        landmark = data["points"]
        landmark = normalize_landmark(landmark, (image.shape[1], image.shape[0]))
        
        # Resize image
        image = cv2.resize(image, (self.input_size))

        # Flip image
        # and update the order of landmark points
        if self.random_flip and random.choice([0, 1]):
            image = cv2.flip(image, 1)

            # Flip landmark
            landmark[:, 0] = 1 - landmark[:, 0]

            # Change the indices of landmark points and visibility
            l = landmark
            landmark = [l[6], l[5], l[4], l[3], l[2], l[1], l[0]]

        # Unnormalize landmark points
        landmark = unnormalize_landmark(landmark, self.input_size)

        if self.augment:
            image, landmark = augment_img(image, landmark)

        # Generate heatmap
        gtmap = None
        if self.output_heatmap:
            gtmap_kps = landmark.copy()
            gtmap_kps[:, :2] = (np.array(gtmap_kps[:, :2]).astype(float)
                        * np.array(self.heatmap_size) / np.array(self.input_size) * np.array(self.heatmap_size)).astype(int)
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
        # if self.output_heatmap:
        #     cv2.imshow("gtmap", gtmap.sum(axis=2))
        # cv2.waitKey(0)

        return image, landmark, gtmap

    @classmethod
    def get_kp_keys(cls):
        keys = ['l_wrist', 'l_ankle', 'l_shoulder',
                'head_top', 'r_shoulder', 'r_ankle',
                'r_wrist']
        return keys

    def generate_gtmap(self, joints, sigma, outres):
        npart = joints.shape[0]
        gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
        for i in range(npart):
            visibility = joints[i, 2]
            if visibility > 0:
                gtmap[:, :, i] = draw_labelmap(gtmap[:, :, i], joints[i, :], sigma)
        return gtmap