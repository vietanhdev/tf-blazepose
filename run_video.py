import argparse
import importlib
import json
import cv2
import numpy as np
from src.utils.heatmap import find_keypoints_from_heatmap
from src.utils.visualizer import visualize_keypoints
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--conf_file', default="config.json",
    help='Configuration file')
parser.add_argument(
    '-m',
    '--model', default="model.h5",
    help='Path to h5 model')
parser.add_argument(
    '-confidence',
    '--confidence',
    default=0.05,
    help='Confidence for heatmap point')
parser.add_argument(
    '-v',
    '--video',
    help='Path to video file')

args = parser.parse_args()

# Webcam
if args.video == "webcam":
    args.video = 0

confth = float(args.confidence)

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Load model
trainer = importlib.import_module("src.trainers.{}".format(config["trainer"]))
model = trainer.load_model(config, args.model)

# Dataloader
datalib = importlib.import_module("src.data_loaders.{}".format(config["data_loader"]))
DataSequence = datalib.DataSequence

cap = cv2.VideoCapture(args.video)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while(True):

    ret, origin_frame = cap.read()

    scale = np.array([float(origin_frame.shape[1]) / config["model"]["im_width"],
                float(origin_frame.shape[0]) / config["model"]["im_height"]], dtype=float)

    img = cv2.resize(origin_frame, (config["model"]["im_width"], config["model"]["im_height"]))
    input_x = DataSequence.preprocess_images(np.array([img]))

    regress_kps, heatmap = model.predict(input_x)
    heatmap_kps = find_keypoints_from_heatmap(heatmap)[0]
    heatmap_kps = np.array(heatmap_kps)

    # Scale heatmap keypoint
    heatmap_stride = np.array([config["model"]["im_width"] / config["model"]["heatmap_width"],
                            config["model"]["im_height"] / config["model"]["heatmap_height"]], dtype=float)
    heatmap_kps[:, :2] = heatmap_kps[:, :2] * scale * heatmap_stride

    # Scale regression keypoint
    regress_kps = regress_kps.reshape((-1, 3))
    regress_kps[:, :2] = regress_kps[:, :2] * np.array([origin_frame.shape[1], origin_frame.shape[0]])

    # Filter heatmap keypoint by confidence
    heatmap_kps_visibility = np.ones((len(heatmap_kps),), dtype=int)
    for i in range(len(heatmap_kps)):
        if heatmap_kps[i, 2] < confth:
            heatmap_kps[i, :2] = [-1, -1]
            heatmap_kps_visibility[i] = 0

    regress_kps_visibility = np.ones((len(regress_kps),), dtype=int)
    for i in range(len(regress_kps)):
        if regress_kps[i, 2] < 0.5:
            regress_kps[i, :2] = [-1, -1]
            regress_kps_visibility[i] = 0

    edges = [[0,1,2,3,4,5,6]]

    draw = origin_frame.copy()
    draw = visualize_keypoints(draw, regress_kps[:, :2], visibility=regress_kps_visibility, edges=edges, point_color=(0, 255, 0), text_color=(255, 0, 0))
    draw = visualize_keypoints(draw, heatmap_kps[:, :2], visibility=heatmap_kps_visibility, edges=edges, point_color=(0, 255, 0), text_color=(0, 0, 255))
    cv2.imshow('Result', draw)

    heatmap = np.sum(heatmap[0], axis=2)
    heatmap = cv2.resize(heatmap, None, fx=3, fy=3)
    heatmap = heatmap * 1.5
    cv2.imshow('Heatmap', heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
