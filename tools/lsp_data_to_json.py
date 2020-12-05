import os
import argparse
import json
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input_file', default="data/lsp_dataset/joints.mat",
    help='Path to lsp annotation file')
parser.add_argument(
    '-f',
    '--image_folder', default="data/lsp_dataset/images",
    help='Image folder')
parser.add_argument(
    '-o',
    '--output_file', default="data/lsp_dataset/labels.json",
    help='Output json file')
args = parser.parse_args()

# Load annotations
annotations = loadmat(args.input_file)
joints = annotations["joints"]
joints_shape = joints.shape
print(joints_shape)
if joints_shape[0] == 3 and joints_shape[1] == 14: # LSP (3, 14, n) -> (n, 14, 3)
    joints = joints.swapaxes(0, 2)
elif joints_shape[0] == 14 and joints_shape[1] == 3: # LSPET (14, 3, n) -> (n, 14, 3)
    joints = joints.swapaxes(0, 2)
    joints = joints.swapaxes(1, 2)

# List image files
images = [i for i in os.listdir(args.image_folder) if i.endswith("jpg")]
images.sort()

# Build new annotations
labels = []
w_img_count = 0
for i in range(len(images)):
    points = joints[i, :, :2]
    visibility = joints[i, :, 2]

    wrong_label = False
    for p in points:
        if p[0] <= 0 or p[1] <= 0:
            wrong_label = True
            break
    if wrong_label:
        w_img_count += 1
        print(w_img_count)
        continue

    label = {"image": images[i], "points": points.tolist(), "visibility": visibility.tolist()}
    labels.append(label)

print("Len: ", len(labels))

with open(args.output_file, "w") as fp:
    json.dump(labels, fp)

