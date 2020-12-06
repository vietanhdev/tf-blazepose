import tensorflow as tf
import numpy as np


@tf.function
def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, kernel, 1, padding='SAME')
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat*keep


@tf.function
def find_keypoints(batch_heatmaps):
    batch, height, width, n_points = tf.shape(batch_heatmaps)[0], tf.shape(
        batch_heatmaps)[1], tf.shape(batch_heatmaps)[2], tf.shape(batch_heatmaps)[3]

    batch_heatmaps = nms(batch_heatmaps)

    flat_tensor = tf.reshape(batch_heatmaps, (batch, -1, n_points))

    # Argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

    # Convert indexes into 2D coordinates
    argmax_x = argmax // width
    argmax_y = argmax % width

    # Shape: batch * 2 * n_points
    batch_keypoints = tf.stack((argmax_x, argmax_y), axis=1)
    # Shape: batch * n_points * 2
    batch_keypoints = tf.transpose(batch_keypoints, [0, 2, 1])

    return batch_keypoints


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
            batch_keypoints_pred = find_keypoints(y_pred)
            batch_keypoints_true = find_keypoints(y_true)
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
