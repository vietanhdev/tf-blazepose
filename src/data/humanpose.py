import os
import numpy as np
import cv2
import random
from tensorflow.keras.utils import Sequence
import math
import random
import json

from .utils import normalize_landmark, unnormalize_landmark
from .augmentation import augment_img

class DataSequence(Sequence):

    def __init__(self, image_folder, label_file, batch_size=8, input_size=(256, 256), shuffle=True, augment=False, random_flip=True, normalize=True):

        self.batch_size = batch_size
        self.input_size = input_size
        self.image_folder = image_folder
        self.random_flip = random_flip
        self.augment = augment
        self.normalize = normalize

        with open(label_file, "r") as fp:
            data = json.load(fp)["labels"]

            # Only keep data which contains human
            self.data = [d for d in data if d.get("contains_person", True)]

        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return math.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        batch_data = self.data[idx *
                               self.batch_size: (1 + idx) * self.batch_size]

        batch_image = []
        batch_landmark = []

        for data in batch_data:

            # Flip 50% of images
            flip = False
            if self.random_flip and random.random() < 0.5:
                flip = True

            # Load and augment data
            image, landmark = self.load_data(
                self.image_folder, data, augment=self.augment, flip=flip)

            batch_image.append(image)
            batch_landmark.append(landmark)

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark)

        batch_image = self.preprocess_images(batch_image)
        batch_landmark = self.preprocess_landmarks(batch_landmark)

        return batch_image, batch_landmark

    def preprocess_images(self, images):
        images = np.array(images, dtype=np.float32)
        images = (images - 127) / 255
        return images

    def preprocess_landmarks(self, landmarks):
        landmark = normalize_landmark(landmarks, self.input_size)

        # Add visibility output
        landmark = landmark.reshape(2, -1)
        visibility = np.zeros((landmark.shape[0], 1))
        landmark = np.hstack((landmark, visibility))
        landmark = landmark.reshape(self.batch_size, -1)

        return landmark

    def load_data(self, img_folder, data, augment=False, flip=False):

        # Load image
        path = os.path.join(img_folder, data["image"])
        img = cv2.imread(path)

        # Normalize landmark for ease of processing
        # After this step, landmark points will have
        # x and y in range of [0, 1]
        landmark = data["points"]
        landmark = normalize_landmark(landmark, (img.shape[1], img.shape[0]))

        # Resize image
        img = cv2.resize(img, (self.input_size))

        # Flip image
        # and update the order of landmark points
        if flip:
            img = cv2.flip(img, 1)

            # Flip landmark
            landmark[:, 0] = 1 - landmark[:, 0]

            # Change the indices of landmark points and visibility
            l = landmark
            landmark = [l[6], l[5], l[4], l[3], l[2], l[1], l[0]]

        # Unnormalize landmark points
        landmark = unnormalize_landmark(landmark, self.input_size)

        if augment:
            img, landmark = augment_img(img, landmark)

        # # Uncomment following lines to write out augmented images for debuging
        # cv2.imwrite("aug_" + str(random.randint(0, 50)) + ".png", img)
        # cv2.waitKey(0)

        # draw = img.copy()
        # landmark = unnormalize_landmark(landmark, self.input_size)
        # for i in range(len(landmark)):
        #     x = int(landmark[i][0])
        #     y = int(landmark[i][1])

        #     draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.circle(draw, (int(x), int(y)), 1, (0,0,255))

        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)

        return img, landmark
