import json
import random
import cv2
import os
import copy
import shutil

data = []
with open("data/pushup/train.json") as fp:
    data1 = json.load(fp)["labels"]
with open("data/pushup/test.json") as fp:
    data2 = json.load(fp)["labels"]
with open("data/pushup/val.json") as fp:
    data3 = json.load(fp)["labels"]

data = [d for d in data if d["contains_person"] and d["is_pushing_up"]]
print(len(data))

mpii1 = [d for d in data1 if d["contains_person"] and not d["is_pushing_up"]]
mpii2 = [d for d in data2 if d["contains_person"] and not d["is_pushing_up"]]
mpii3 = [d for d in data3 if d["contains_person"] and not d["is_pushing_up"]]

print(len(mpii1))
print(len(mpii2))
print(len(mpii3))
exit(0)

kk = {}
for d in data:
    d["video"] = d["image"].split("_")[0]
    kk[d["image"]] = d

new_data = []
for k in kk.values():
    new_data.append(k)

data = new_data
random.shuffle(data)

for d in data:
    with open(os.path.join("labels", "{}.json".format(d["image"])), "w") as fp:
        json.dump(d["points"], fp)

# for d in data:
#     # if d["video"] == "317":
#     #     img = cv2.imread(os.path.join("data/pushup/images", d["image"]))
#     #     cv2.imshow("Image", img)
#     #     cv2.waitKey(0)
#     shutil.copy(os.path.join("data/pushup/images", d["image"]),
#     os.path.join("data/pushup/new_images", d["image"]))

# videos = {}
# for d in data:
#     if d["video"] not in videos:
#         videos[d["video"]] = 1
#     else:
#         videos[d["video"]] += 1

# video_img_count = []
# for video, img_count in videos.items():
#     print("{}:{}".format(video, img_count), end="; ")

# print("We have {} videos", len(videos.keys()))