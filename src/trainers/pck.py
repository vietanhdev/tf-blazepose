import tensorflow as tf
import numpy as np

from ..utils.heatmap import find_keypoints_from_heatmap


@tf.function
def calc_pck(batch_keypoints_true, batch_keypoints_pred, ref_point_pair=(3, 5), thresh=0.2):
    ref_distance = tf.math.reduce_euclidean_norm(
        batch_keypoints_true[:, ref_point_pair[0], :] - batch_keypoints_true[:, ref_point_pair[1], :], axis=1, keepdims=True)
    error = tf.math.reduce_euclidean_norm(
        batch_keypoints_pred - batch_keypoints_true, axis=2)
    wrong_matrix = tf.cast(error, tf.float32) > tf.cast(
        ref_distance, tf.float32) * thresh
    n_wrongs = tf.reduce_sum(tf.cast(wrong_matrix, tf.float32))
    return tf.cast(n_wrongs, tf.float32), tf.cast(tf.size(batch_keypoints_true), tf.float32)

class PCK(tf.keras.metrics.Metric):

    def __init__(self, name='pck', ref_point_pair=(3, 5), thresh=0.2, **kwargs):
        super(PCK, self).__init__(name=name, **kwargs)
        self.ref_point_pair = ref_point_pair
        self.thresh = 0.2
        self.n_wrongs = self.add_weight(name='n_wrongs', initializer='zeros')
        self.n_total = self.add_weight(name='n_total', initializer='zeros')

    def reset_states(self):
        self.n_wrongs.assign(0)
        self.n_total.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):

        if len(tf.shape(y_true)) == 4:  # Heatmap
            batch_keypoints_pred = find_keypoints_from_heatmap(y_pred)[:, :, :2]
            batch_keypoints_true = find_keypoints_from_heatmap(y_true)[:, :, :2]
        elif len(tf.shape(y_true)) == 2:  # Regression
            batch_keypoints_pred = tf.reshape(
                y_pred, (tf.shape(y_pred)[0], -1, 3))[:, :, :2]
            batch_keypoints_true = tf.reshape(
                y_true, (tf.shape(y_true)[0], -1, 3))[:, :, :2]
        else:
            tf.print("Error: Wrong PCK input shape.")
            exit(0)
        n_wrongs, n_points = calc_pck(
            batch_keypoints_true, batch_keypoints_pred, ref_point_pair=self.ref_point_pair, thresh=self.thresh)
        self.n_wrongs.assign_add(n_wrongs)
        self.n_total.assign_add(n_points)

    def result(self):
        return (self.n_total - self.n_wrongs) / (self.n_total + 1e-5)
