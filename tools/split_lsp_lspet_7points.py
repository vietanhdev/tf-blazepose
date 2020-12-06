"""
This script is used to concatenate datasets into a bigger dataset
and split them into training, validation and test sets.
"""
import os
import pathlib
import json
import shutil
import random
import cv2
import numpy as np

from src.utils.pre_processing import calculate_bbox_from_keypoints, square_crop_with_keypoints
from src.data_loaders.augmentation_utils import random_occlusion


class DatasetCreator:
    def __init__(self, root_folder):

        if os.path.exists(root_folder):
            print("Folder existed! Please choose a non-existed path. {}".format(root_folder))
            exit(0)

        self.root_folder = root_folder
        self.image_folder = os.path.join(root_folder, "images")
        pathlib.Path(self.root_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.image_folder).mkdir(parents=True, exist_ok=True)


    def save_label(self, file_name, labels):
        with open(os.path.join(self.root_folder, file_name), "w") as fp:
            json.dump(labels, fp)

    def add_set(self, image_folder, label_file, n_train, n_val, n_test, copy_images=True, shuffle=True):
        """Add subset into this dataset

        Args:
            image_folder (str): Path to image folder of subset
            label_file (str): Path to annotation file of subset
            n_train (int): Number of samples for training
            n_val (int): Number of samples for validation
            n_test (int): Number of samples for testing
        """

        with open(label_file, "r") as fp:
            labels = json.load(fp)

        # Use all data?
        assert (len(labels) == n_train + n_val + n_test)

        if shuffle:
            random.seed(42)
            random.shuffle(labels)

        for label in labels:

            # Copy images
            if copy_images:

                # Rename image if duplicated
                if os.path.exists(os.path.join(self.image_folder, label["image"])):
                    new_name = label["image"]
                    filename, file_extension = os.path.splitext(label["image"])
                    extended_number = 2
                    while True:
                        new_name = "{}_ext{}{}".format(filename, extended_number, file_extension)
                        if os.path.exists(os.path.join(self.image_folder, new_name)):
                            extended_number += 1
                        else:
                            break
                    shutil.copy(
                        os.path.join(image_folder, label["image"]),
                        os.path.join(self.image_folder, new_name),
                    )
                    label["image"] = new_name
                else:
                    shutil.copy(
                        os.path.join(image_folder, label["image"]),
                        os.path.join(self.image_folder, label["image"]),
                    )
                    pass

            label["bbox"] = calculate_bbox_from_keypoints(label["points"])
            label["bbox"] = np.array(label["bbox"]).astype(int).tolist()
        
        # # Visualize
        # for label in labels:
        #     image = cv2.imread(os.path.join(self.image_folder, label["image"]))

        #     draw = image.copy()
        #     for i, p in enumerate(label["points"]):
        #         x, y = p
        #         color = (0, 0, 255) if int(label["visibility"][i]) else (255, 0, 0)
        #         draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
        #         draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #     p1 = tuple(label["bbox"][0])
        #     p2 = tuple(label["bbox"][1])
        #     draw = cv2.rectangle(draw, p1, p2, (0,0,255), 2)
        #     cv2.imshow("Image", draw)
        #     # cv2.waitKey(0)

        #     cropped_image, keypoints = square_crop_with_keypoints(image, label["bbox"], label["points"], "random")
        #     draw = cropped_image.copy()
        #     for i, p in enumerate(keypoints):
        #         x, y = p
        #         color = (0, 0, 255) if int(label["visibility"][i]) else (255, 0, 0)
        #         draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
        #         draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        #     cv2.imshow("Square cropped", draw)
        #     # cv2.waitKey(0)

        #     # Test random occlusion
        #     cropped_image, visibility = random_occlusion(cropped_image, keypoints, visibility=None, rect_ratio=((0.2, 0.5), (0.2, 0.5)))
        #     draw = cropped_image.copy()
        #     for i, p in enumerate(keypoints):
        #         print(i)
        #         print(visibility[i])
        #         x, y = p
        #         color = (0, 0, 255) if int(visibility[i]) else (255, 0, 0)
        #         draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
        #         draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        #     cv2.imshow("Random occlusion", draw)
        #     cv2.waitKey(0)

        for i, label in enumerate(labels):
            ps = label["points"]
            labels[i]["points"] = [ps[6], ps[7], ps[8], ps[13], ps[9], ps[10], ps[11]]
            vs = label["visibility"]
            labels[i]["visibility"] = [vs[6], vs[7], vs[8], vs[13], vs[9], vs[10], vs[11]]


        self.save_label("train.json", labels[:n_train])
        self.save_label("val.json", labels[n_train:n_train+n_val])
        self.save_label("test.json", labels[n_train+n_val:])

dataset = DatasetCreator("data/lsp_lspet_7points")
dataset.add_set("data/lsp_dataset/images", "data/lsp_dataset/labels.json", 1800, 100, 100)
dataset.add_set("data/lspet_dataset/images", "data/lspet_dataset/labels.json", 3739, 100, 100)
