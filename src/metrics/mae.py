import tensorflow as tf
import numpy as np

from ..utils.heatmap import find_keypoints_from_heatmap


@tf.function
def calc_mae(batch_keypoints_true, batch_keypoints_pred, keypoint_thresh=0.1):

    mask = tf.greater(batch_keypoints_true[:, :, 2], keypoint_thresh)
    tf.boolean_mask(batch_keypoints_true[:, :, 0], mask)
    tf.boolean_mask(batch_keypoints_true[:, :, 1], mask)

    mask = tf.greater(batch_keypoints_pred[:, :, 2], keypoint_thresh)
    tf.boolean_mask(batch_keypoints_pred[:, :, 0], mask)
    tf.boolean_mask(batch_keypoints_pred[:, :, 1], mask)

    error = tf.abs(batch_keypoints_pred[:, :, :2] - batch_keypoints_true[:, :, :2])
    n_points = tf.cast(tf.reduce_prod(tf.shape(error)), tf.float32)
    error = tf.reduce_sum(tf.cast(error, tf.float32))

    return error, n_points


def get_mae_metric():

    class MAE(tf.keras.metrics.Metric):

        def __init__(self, name='mae', **kwargs):
            super(MAE, self).__init__(name=name, **kwargs)
            self.total_error = self.add_weight(name='total_error', initializer='zeros')
            self.n_total = self.add_weight(name='n_total', initializer='zeros')

        def reset_states(self):
            self.total_error.assign(0)
            self.n_total.assign(0)

        def update_state(self, y_true, y_pred, sample_weight=None):

            keypoint_thresh = 0.0
            if len(tf.shape(y_true)) == 4:  # Heatmap
                batch_keypoints_pred = find_keypoints_from_heatmap(y_pred, normalize=True)
                batch_keypoints_true = find_keypoints_from_heatmap(y_true, normalize=True)
                keypoint_thresh = 0.1
            elif len(tf.shape(y_true)) == 2:  # Regression
                batch_keypoints_pred = tf.reshape(
                    y_pred, (tf.shape(y_pred)[0], -1, 3))
                batch_keypoints_true = tf.reshape(
                    y_true, (tf.shape(y_true)[0], -1, 3))
                keypoint_thresh = 0.5
            else:
                tf.print("Error: Wrong MAE input shape.")
                exit(0)
            error, n_points = calc_mae(
                batch_keypoints_true, batch_keypoints_pred, keypoint_thresh=keypoint_thresh)
            self.total_error.assign_add(error)
            self.n_total.assign_add(n_points)

        def result(self):
            return self.total_error / (self.n_total + 1e-5)

    return MAE
