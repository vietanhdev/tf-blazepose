import tensorflow as tf
import numpy as np
from ..utils.heatmap_process import post_process_heatmap

class PCKS(tf.keras.callbacks.Callback):
    def on_train_begin(self, model, logs={}):
        self._data = []
        self.model = model

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_rocauc': roc_auc_score(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data