import os
import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from ..model_type import ModelType
from ..models.keypoint_detection.blazepose import BlazePose
from ..data.humanpose import DataSequence


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
    pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)
    mc = ModelCheckpoint(filepath=os.path.join(
        model_folder_path, "model_ep{epoch:03d}.h5"), save_weights_only=True, save_format="h5", verbose=1)

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

def test(config, model_path):
    """Test trained model

    Args:
        config (dict): Model configuration
        model (str): Path to h5 model to be tested
    """

    train_config = config["train"]
    model_config = config["model"]

    # Initialize model and load weights
    model = BlazePose(
        model_config["num_joints"], ModelType(model_config["model_type"])).build_model()
    model.compile(loss="binary_crossentropy", metrics=["mean_absolute_error"])
    model.load_weights(model_path)

    # Load data
    model_config = config["model"]
    test_dataset = DataSequence(
        config["data"]["test_images"],
        config["data"]["test_labels"],
        batch_size=1,
        input_size=(model_config["im_width"], model_config["im_height"]),
        shuffle=False, augment=False, random_flip=False)

    # Test model
    model.evaluate(test_dataset, verbose=1)
