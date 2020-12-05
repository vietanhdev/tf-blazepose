import os
import argparse
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--image_folder', default="data/lsp_dataset/images",
    help='Image folder')
parser.add_argument(
    '-l',
    '--labels', default="data/lsp_dataset/labels.json",
    help='Label/Annotation file')
args = parser.parse_args()

with open(args.labels, "r") as fp:
    labels = json.load(fp)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
for label in labels:
    image_name = label["image"]
    points = label["points"]
    visibility = label["visibility"]
    image = cv2.imread(os.path.join(args.image_folder, image_name))
    for i, p in enumerate(points):
        x, y = p
        color = (0, 0, 255) if int(visibility[i]) else (255, 0, 0)
        cv2.circle(image, center=(int(x), int(y)), color=(255, 0, 0), radius=1, thickness=2)
        image = cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

