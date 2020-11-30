import os
import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from ..model_type import ModelType
from ..models.keypoint_detection.blazepose import BlazePose
from ..data.mpii_datagen import MPIIDataGen
from .losses import focal_tversky
from .pcks import PCKS

def train(config):
    """Train model

    Args:
        config (dict): Training configuration from configuration file
    """

    train_config = config["train"]
    model_config = config["model"]

    # Initialize model
    model = BlazePose(
        model_config["num_joints"], ModelType(model_config["model_type"])).build_model()

    loss_function = train_config["loss_function"]
    if loss_function == "focal_tversky":
        loss_function = focal_tversky

    model.compile(optimizer=tf.optimizers.Adam(train_config["learning_rate"]),
                  loss=loss_function, metrics=[tf.keras.metrics.MeanIoU(num_classes=16)])

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
    pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)
    mc = ModelCheckpoint(filepath=os.path.join(
        model_folder_path, "model_ep{epoch:03d}.h5"), save_weights_only=True, save_format="h5", verbose=1)

    # Load data
    train_dataset = DataSequence(
        config["data"]["train_images"],
        config["data"]["train_labels"],
        batch_size=train_config["train_batch_size"],
        input_size=(model_config["im_width"], model_config["im_height"]),
        heatmap_size=(model_config["heatmap_width"], model_config["heatmap_height"]),
        shuffle=True, augment=True, random_flip=True)
    val_dataset = DataSequence(
        config["data"]["val_images"],
        config["data"]["val_labels"],
        batch_size=train_config["val_batch_size"],
        input_size=(model_config["im_width"], model_config["im_height"]),
        heatmap_size=(model_config["heatmap_width"], model_config["heatmap_height"]),
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


def load_model(config, model_path):
    """Load pretrained model

    Args:
        config (dict): Model configuration
        model (str): Path to h5 model to be tested
    """

    model_config = config["model"]

    # Initialize model and load weights
    model = BlazePose(
        model_config["num_joints"], ModelType(model_config["model_type"])).build_model()
    model.compile()
    model.load_weights(model_path)

    return model