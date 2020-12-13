
import tensorflow.keras.backend as K
K.set_learning_phase(0)
import tensorflow as tf
import keras2onnx
from tensorflow.keras.models import load_model

MODEL_PATH = ""
model = load_model(MODEL_PATH)
submodel = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer("joints").outputs)
submodel._name = "blazepose_heatmap_v1"
print(submodel.summary())
onnx_model = keras2onnx.convert_keras(submodel, submodel.name)

file = open("blazepose_heatmap_v1.1.onnx", "wb")
file.write(onnx_model.SerializeToString())
file.close()