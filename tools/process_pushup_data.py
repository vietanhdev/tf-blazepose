import os
import pathlib
import json
import shutil
import random
import cv2
import numpy as np
from src.utils.pre_processing import calculate_bbox_from_keypoints, square_crop_with_keypoints, random_occlusion


label_files = list(os.listdir("data/pushup/labels"))
label_files = [l for l in label_files if l.endswith("json")]

labels = []
for lf in label_files:
    with open(os.path.join("data/pushup/labels", lf)) as fp:
        points = json.load(fp)
        if len(points) != 7:
            continue
        label = {
            "video": int(lf.split("_")[0]),
            "image": lf.replace(".json", ""),
            "points": points,
            "visibility": [1,1,1,1,1,1,1]
        }
        image = cv2.imread(os.path.join("data/pushup/images", label["image"]))
        label["bbox"] = [[0,0],[image.shape[1],image.shape[0]]]
        label["bbox"] = np.array(label["bbox"]).astype(int).tolist()
        labels.append(label)

# for label in labels:
#     # Visualize
#     for label in labels:
#         image = cv2.imread(os.path.join("data/pushup/images", label["image"]))

#         draw = image.copy()
#         for i, p in enumerate(label["points"]):
#             x, y = p
#             color = (0, 0, 255) if int(label["visibility"][i]) else (255, 0, 0)
#             draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
#             draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 255, 0), 1, cv2.LINE_AA)
#         p1 = tuple(label["bbox"][0])
#         p2 = tuple(label["bbox"][1])
#         draw = cv2.rectangle(draw, p1, p2, (0,0,255), 2)
#         cv2.imshow("Image", draw)
#         # cv2.waitKey(0)

#         cropped_image, keypoints = square_crop_with_keypoints(image, label["bbox"], label["points"], "random")
#         draw = cropped_image.copy()
#         for i, p in enumerate(keypoints):
#             x, y = p
#             color = (0, 0, 255) if int(label["visibility"][i]) else (255, 0, 0)
#             draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
#             draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 255, 0), 1, cv2.LINE_AA)

#         cv2.imshow("Square cropped", draw)
#         # cv2.waitKey(0)

#         # Test random occlusion
#         cropped_image, visibility = random_occlusion(cropped_image, keypoints, visibility=None, rect_ratio=((0.2, 0.5), (0.2, 0.5)))
#         draw = cropped_image.copy()
#         for i, p in enumerate(keypoints):
#             print(i)
#             print(visibility[i])
#             x, y = p
#             color = (0, 0, 255) if int(visibility[i]) else (255, 0, 0)
#             draw = cv2.circle(draw, center=(int(x), int(y)), color=color, radius=1, thickness=2)
#             draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 255, 0), 1, cv2.LINE_AA)

#         cv2.imshow("Random occlusion", draw)
#         cv2.waitKey(0)

print("n_labels:", len(labels))

# for label in labels:
#     label.pop("video", None)

train_labels = [l for l in labels if l["video"] < 480]
# print("train_labels", len(train_labels))
# train_videos = set(l["video"] for l in train_labels)
# print("train_videos", len(train_videos))

val_labels = [l for l in labels if l["video"] >= 480 and l["video"] < 550]
# print("val_labels", len(val_labels))
# val_videos = set(l["video"] for l in val_labels)
# print("val_videos", len(val_videos))

test_labels = [l for l in labels if l["video"] >= 550]
# print("test_labels", len(test_labels))
# test_videos = set(l["video"] for l in test_labels)
# print("test_videos", len(test_videos))

with open("data/pushup/train.json", "w") as fp:
    json.dump(train_labels, fp)

with open("data/pushup/val.json", "w") as fp:
    json.dump(val_labels, fp)

with open("data/pushup/test.json", "w") as fp:
    json.dump(test_labels, fp)


