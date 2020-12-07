import json
import os
import pathlib
import json
import shutil
import random
import cv2
import numpy as np

from src.utils.pre_processing import calculate_bbox_from_keypoints, square_crop_with_keypoints
from src.data_loaders.augmentation_utils import random_occlusion

image_folder = "data/mpii/images"
jsonfile = "data/mpii_annotations.json"

# load train or val annotation
with open(jsonfile) as anno_file:
    anno = json.load(anno_file)

val_anno, train_anno = [], []
for idx, val in enumerate(anno):
    if val['isValidation'] == True:
        val_anno.append(anno[idx])
    else:
        train_anno.append(anno[idx])

anno = train_anno + val_anno

count = 0
labels = []
for a in anno:

    if a["numOtherPeople"] != 0:
        continue

    l = {"image": a["img_paths"]}
    points = np.array(a["joint_self"])
    l["points"] = points[:, :2].tolist()
    l["visibility"] = points[:, 2].tolist()

    for p in l["points"]:
        if p[0] == 0 and p[1] == 0:
            p[0] = -1
            p[1] = -1


    inside_points = [p for p in l["points"] if p[0] != -1 or p[1] != -1]
    l["bbox"] = calculate_bbox_from_keypoints(inside_points)
    l["bbox"] = np.array(l["bbox"]).astype(int).tolist()

    # Crop image
    image = cv2.imread(os.path.join(image_folder, l["image"]))
    image, keypoints = square_crop_with_keypoints(image, l["bbox"], l["points"], "random")
    l["points"] = keypoints.tolist()

    # BBox again
    inside_points = [p for p in l["points"] if p[0] != -1 or p[1] != -1]
    l["bbox"] = calculate_bbox_from_keypoints(inside_points)
    l["bbox"] = np.array(l["bbox"]).astype(int).tolist()

    l["image"] = "mpii_crop_{}.png".format(count)
    count += 1

    # cv2.imwrite(os.path.join("data/mpii/images2", l["image"]), image)

    # draw = image.copy()
    # for i, p in enumerate(l["points"]):
    #     x, y = p
    #     color = (0, 0, 255) if int(l["visibility"][i]) else (255, 0, 0)
    #     draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
    #     draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #     draw = cv2.rectangle(draw, tuple(l["bbox"][0]), tuple(l["bbox"][1]), (0, 0, 255), 5)
    # cv2.imshow("Image", draw)
    # cv2.waitKey(0)


    labels.append(l)
    
# # Visualize
# for label in labels:
#     image = cv2.imread(os.path.join(image_folder, label["image"]))

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

def save_label(file_name, labels):
    with open(os.path.join("data/mpii/", file_name), "w") as fp:
        json.dump(labels, fp)

with open("data/mpii/labels.json", "w") as anno_file:
    json.dump(labels, anno_file)


n_train = 9503
n_val = 1000
save_label("train.json", labels[:n_train])
save_label("val.json", labels[n_train:n_train+n_val])
save_label("test.json", labels[n_train+n_val:])

print(len(labels))

# with open("data/mpii/train.json", "w") as anno_file:
#     json.dump(train_anno, anno_file)
    
# with open("data/mpii/val.json", "w") as anno_file:
#     json.dump(val_anno, anno_file)