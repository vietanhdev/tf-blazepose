import argparse
import importlib
import json
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
args = parser.parse_args()

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

trainer = importlib.import_module("src.trainers.{}".format(config["trainer"]))
trainer.test(config, args.model)
