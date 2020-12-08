import argparse
import importlib
import json

import cv2
import numpy as np
import tensorflow as tf

from src.utils.heatmap import find_keypoints_from_heatmap
from src.utils.visualizer import visualize_keypoints

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
    help='Path to video file')
parser.add_argument(
    '-v',
    '--video',
    help='Confidence for heatmap point')
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


CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("trained_models/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("trained_models/yolov3-tiny.weights", "trained_models/yolov3-tiny.cfg")

yolo = cv2.dnn_DetectionModel(net)
yolo.setInputParams(size=(416, 416), scale=1/255)


def find_interested_area(image, bbox):
    bbox_width = bbox[1][0] - bbox[0][0]
    bbox_height = bbox[1][1] - bbox[0][1]
    im_height, im_width = image.shape[:2]

    if bbox_width > bbox_height: # Padding on y-axis
        pad = int((bbox_width - bbox_height) / 2)
        bbox[0][1] -= pad
        bbox[1][1] = bbox[0][1] + bbox_width
    elif bbox_height > bbox_width: # Padding on x-axis
        pad = int((bbox_height - bbox_width) / 2)
        bbox[0][0] -= pad
        bbox[1][0] = bbox[0][0] + bbox_height

    bbox[0][0] = max(bbox[0][0], 0)
    bbox[0][1] = max(bbox[0][1], 0)
    bbox[1][0] = min(bbox[1][0], im_width - 1)
    bbox[1][1] = min(bbox[1][1], im_height - 1)
    return bbox

cap = cv2.VideoCapture(args.video)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while(True):

    ret, origin_frame = cap.read()

    scale = np.array([float(origin_frame.shape[1]) / config["model"]["im_width"],
                float(origin_frame.shape[0]) / config["model"]["im_height"]], dtype=float)

    img = cv2.resize(origin_frame, (config["model"]["im_width"], config["model"]["im_height"]))
    draw = origin_frame.copy()
   
    
    yolo_input = cv2.resize(origin_frame, (416, 416))
    classes, scores, boxes = yolo.detect(yolo_input, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        box = (np.array(box).reshape((2, 2)) * np.array([draw.shape[1] / yolo_input.shape[1],
                                            draw.shape[0] / yolo_input.shape[0]])).astype(int).flatten().tolist()
        cv2.rectangle(draw, box, color, 2)
        cv2.putText(draw, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    keypoint_model_input = img
    bbox = [[0,0],[0,0]]
    if len(scores) > 0:
        bboxes_with_scores = list(zip(scores, boxes))
        bboxes_with_scores.sort()
        bbox = bboxes_with_scores[-1][1]
        bbox = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]
        bbox = find_interested_area(keypoint_model_input, bbox)

        viz = keypoint_model_input.copy()
        cv2.rectangle(viz, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 2)
        cv2.imshow("viz", viz)
        cv2.waitKey(1)

        # keypoint_model_input = keypoint_model_input[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    input_x = DataSequence.preprocess_images(np.array([keypoint_model_input]))
    regress_kps, heatmap = model.predict(input_x)
    heatmap_kps = find_keypoints_from_heatmap(heatmap)[0]
    heatmap_kps = np.array(heatmap_kps)

    # Scale heatmap keypoint
    heatmap_stride = np.array([config["model"]["im_width"] / config["model"]["heatmap_width"],
                            config["model"]["im_height"] / config["model"]["heatmap_height"]], dtype=float)
    heatmap_kps[:, :2] = heatmap_kps[:, :2] * scale * heatmap_stride

    regress_kps[:, 0] = regress_kps[:, 0] - bbox[0][0]
    regress_kps[:, 1] = regress_kps[:, 1] - bbox[0][1]
    heatmap_kps[:, 0] = heatmap_kps[:, 0] - bbox[0][0]
    heatmap_kps[:, 1] = heatmap_kps[:, 1] - bbox[0][1]

    # Filter heatmap keypoint by confidence
    for i in range(len(heatmap_kps)):
        if heatmap_kps[i, 2] < confth:
            heatmap_kps[i, :2] = [0,0]

    # Scale regression keypoint
    regress_kps = regress_kps.reshape((-1, 3))
    regress_kps[:, :2] = regress_kps[:, :2] * np.array([origin_frame.shape[1], origin_frame.shape[0]])

    draw = visualize_keypoints(draw, regress_kps[:, :2], point_color=(0, 255, 0), text_color=(255, 0, 0))
    draw = visualize_keypoints(draw, heatmap_kps[:, :2], point_color=(0, 255, 0), text_color=(0, 0, 255))
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
