import argparse
import importlib
import json
import cv2
import numpy as np
from src.trainers.blazepose_trainer import load_model
from src.utils.heatmap_process import post_process_heatmap
from src.data.mpii import DataSequence
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

def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y, _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)
    return cvmat

cap = cv2.VideoCapture(args.video)
while(True):
    # Capture frame-by-frame
    ret, origin_frame = cap.read()

    scale = (origin_frame.shape[0] * 1.0 / 256, origin_frame.shape[1] * 1.0 / 256)

    frame = origin_frame
    imgdata = cv2.resize(frame, (256, 256))
    input_x = DataSequence.preprocess_images(np.array([imgdata]))

    regress_kps, out = model.predict(input_x)
    kps = post_process_heatmap(out[0, :, :, :])

    kp_keys = list(map(str, range(len(kps))))
    mkps = list()
    for i, _kp in enumerate(kps):
        _conf = _kp[2]
        mkps.append((_kp[0] * scale[1] * 2, _kp[1] * scale[0] * 2, _conf))
    cvmat = render_joints(origin_frame, mkps, confth)

    # Draw regressed keypoints
    for p in regress_kps.reshape((-1, 3)):
        x = int(p[0] * cvmat.shape[1])
        y = int(p[1] * cvmat.shape[0])
        cv2.circle(cvmat, center=(x, y), color=(0, 0, 255), radius=7, thickness=2)
    cv2.imshow('frame', cvmat)

    out = np.sum(out[0], axis=2)
    out = cv2.resize(out, None, fx=3, fy=3)
    out = out * 1.5
    cv2.imshow('heatmap', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()