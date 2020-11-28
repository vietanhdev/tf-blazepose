import argparse
import importlib
import json
import cv2
import numpy as np
from src.trainers.blazepose_heatmap import load_model
from src.data.data_process import normalize
from src.utils.heatmap_process import post_process_heatmap
from src.data.mpii_datagen import MPIIDataGen
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
    '-v',
    '--video',
    help='Path to video file')
args = parser.parse_args()

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


confth = 0.5


cap = cv2.VideoCapture(args.video)
while(True):
    # Capture frame-by-frame
    ret, origin_frame = cap.read()

    scale = (origin_frame.shape[0] * 1.0 / 256, origin_frame.shape[1] * 1.0 / 256)

    frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    imgdata = cv2.resize(frame, (256, 256))
    mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
    imgdata = normalize(imgdata, mean)
    input_x = imgdata[np.newaxis, :, :, :]

    out = model.predict(input_x)

    kps = post_process_heatmap(out[0, :, :, :])

    ignore_kps = ['plevis', 'thorax', 'head_top']
    kp_keys = MPIIDataGen.get_kp_keys()
    mkps = list()
    for i, _kp in enumerate(kps):
        if kp_keys[i] in ignore_kps:
            _conf = 0.0
        else:
            _conf = _kp[2]
        mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))
    cvmat = render_joints(origin_frame, mkps, confth)

    cv2.imshow('frame', cvmat)

    out = np.sum(out[0], axis=2)
    cv2.imshow('heatmap', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()