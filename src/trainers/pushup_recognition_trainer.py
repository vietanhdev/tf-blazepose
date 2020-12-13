import os
import pathlib
import importlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from ..models import ModelCreator

from .losses import euclidean_distance_loss, focal_tversky, focal_loss, get_huber_loss
from ..metrics.f1 import F1_Score, Recall, Precision

def train(config):
    """Train model

    Args:
        config (dict): Training configuration from configuration file
    """

    train_config = config["train"]
    test_config = config["test"]
    model_config = config["model"]

    # Dataloader
    datalib = importlib.import_module("src.data_loaders.{}".format(config["data_loader"]))
    DataSequence = datalib.DataSequence

    # Initialize model
    model = ModelCreator.create_model(model_config["model_type"])

    print(model.summary())
    model.compile(optimizer=tf.optimizers.Adam(train_config["learning_rate"]),
                  loss=train_config["loss"], metrics=[F1_Score(), Recall(), Precision()])


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
        model_folder_path, "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)

    # Load data
    train_dataset = DataSequence(
        config["data"]["train_images"],
        config["data"]["train_labels"],
        batch_size=train_config["train_batch_size"],
        input_size=(model_config["im_width"], model_config["im_height"]),
        shuffle=True, augment=True, random_flip=True, random_rotate=True,
        random_scale_on_crop=True)
    val_dataset = DataSequence(
        config["data"]["val_images"],
        config["data"]["val_labels"],
        batch_size=train_config["val_batch_size"],
        input_size=(model_config["im_width"], model_config["im_height"]),
        shuffle=False, augment=False, random_flip=False, random_rotate=False,random_scale_on_crop=False)

    # Initial epoch. Use when continue training
    initial_epoch = train_config.get("initial_epoch", 0)

    # Train
    model.fit(train_dataset,
              epochs=train_config["nb_epochs"],
              steps_per_epoch=len(train_dataset),
              validation_data=val_dataset,
              validation_steps=len(val_dataset),
              callbacks=[tb, mc],
              initial_epoch=initial_epoch,
              verbose=1)


def load_model(config, model_path):
    """Load pretrained model

    Args:
        config (dict): Model configuration
        model (str): Path to h5 model to be tested
    """

    model_config = config["model"]

    # Initialize model and load weights
    model = ModelCreator.create_model(model_config["model_type"])
    model.compile()
    model.load_weights(model_path)

    return model