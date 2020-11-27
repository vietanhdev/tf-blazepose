import os
import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model

from ..model_phase import ModelPhase
from ..models.keypoint_detection.blazepose import BlazePose
from ..data.humanpose import DataSequence


def train(config):
    """Train model

    Args:
        config (dict): Training configuration from configuration file
    """

    train_config = config["train"]

    # Initialize model
    model_config = config["model"]
    model = BlazePose(
        model_config["num_joints"], ModelPhase(model_config["model_phase"])).build_model()
    model.compile(optimizer=tf.optimizers.Adam(train_config["learning_rate"]),
                  loss="binary_crossentropy")

    # Load pretrained model
    if train_config["load_weights"]:
        print("Loading model weights: " +
              train_config["pretrained_weights_path"])
        model.load_weights(train_config["pretrained_weights_path"])

    # Create experiment folder
    exp_path = os.path.join("experiments/{}".format(config["experiment_name"]))
    pathlib.Path(exp_path).mkdir(parents=True, exist_ok=True)

    # Define the callbacks
    tb_log_path = os.path.join(exp_path, "tb_logs")
    tb = TensorBoard(log_dir=tb_log_path, write_graph=True)
    model_folder_path = os.path.join(exp_path, "models")
    mc = ModelCheckpoint(filepath=os.path.join(
        model_folder_path, "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=2)

    # Load data
    train_dataset = DataSequence(
        config["data"]["train_images"],
        config["data"]["train_labels"],
        batch_size=train_config["train_batch_size"],
        input_size=(model_config["im_width"], model_config["im_height"]),
        shuffle=True, augment=True, random_flip=True)
    val_dataset = DataSequence(
        config["data"]["val_images"],
        config["data"]["val_labels"],
        batch_size=train_config["val_batch_size"],
        input_size=(model_config["im_width"], model_config["im_height"]),
        shuffle=False, augment=False, random_flip=False)

    # Train
    model.fit(train_dataset,
              epochs=train_config["nb_epochs"],
              steps_per_epoch=len(train_dataset),
              validation_data=val_dataset,
              validation_steps=len(val_dataset),
              callbacks=[tb, mc],
              verbose=1
              )
