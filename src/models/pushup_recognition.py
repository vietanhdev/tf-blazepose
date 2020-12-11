import tensorflow as tf
from tensorflow.keras.models import Model

class PushUpRecognition():

    @staticmethod
    def build_model():

        input_x = tf.keras.layers.Input(shape=(224, 224, 3))
        x = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet',
                                    alpha=0.5)(input_x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        return Model(inputs=input_x, outputs=x)