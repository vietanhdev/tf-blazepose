"""
This script is used to concatenate datasets into a bigger dataset
and split them into training, validation and test sets.
"""
import os
import pathlib
import json
import shutil
import random

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

        for i, label in enumerate(labels):

            ps = label["points"]
            labels[i]["points"] = [ps[6], ps[7], ps[8], ps[13], ps[9], ps[10], ps[11]]
            vs = label["visibility"]
            labels[i]["visibility"] = [vs[6], vs[7], vs[8], vs[13], vs[9], vs[10], vs[11]]

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
                    labels[i]["image"] = new_name
                else:
                    shutil.copy(
                        os.path.join(image_folder, label["image"]),
                        os.path.join(self.image_folder, label["image"]),
                    )

        self.save_label("train.json", labels[:n_train])
        self.save_label("val.json", labels[n_train:n_train+n_val])
        self.save_label("test.json", labels[n_train+n_val:])

dataset = DatasetCreator("data/lsp_lspet_7points")
dataset.add_set("data/lsp_dataset/images", "data/lsp_dataset/labels.json", 1600, 200, 200)
dataset.add_set("data/lspet_dataset/images", "data/lspet_dataset/labels.json", 9000, 500, 500)
