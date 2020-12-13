from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np
import tensorflow as tf


def gen_point_heatmap(img, pt, sigma, type='Gaussian'):
    """Draw label map for 1 point

    Args:
        img: Input image
        pt: Point in format (x, y)
        sigma: Sigma param in Gaussian or Cauchy kernel
        type (str, optional): Type of kernel used to generate heatmap. Defaults to 'Gaussian'.

    Returns:
        np.array: Heatmap image
    """
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def gen_gt_heatmap(keypoints, sigma, heatmap_size):
    """Generate groundtruth heatmap

    Args:
        keypoints: Keypoints in format [[x1, y1], [x2, y2], ...]
        sigma: Sigma param in Gaussian
        heatmap_size: Heatmap size in format (width, height)

    Returns:
        Generated heatmap
    """
    npart = keypoints.shape[0]
    gtmap = np.zeros(shape=(heatmap_size[1], heatmap_size[0], npart), dtype=float)
    for i in range(npart):
        if keypoints[i, 0] == 0 and keypoints[i, 1] == 0:
            continue
        is_visible = True
        if len(keypoints[0]) > 2:
            visibility = keypoints[i, 2]
            if visibility <= 0:
                is_visible = False
        gtmap[:, :, i] = gen_point_heatmap(
            gtmap[:, :, i], keypoints[i, :], sigma)
        if not is_visible:
            gtmap[:, :, i] *= 0.5
    return gtmap


@tf.function
def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, kernel, 1, padding='SAME')
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat*keep


@tf.function
def find_keypoints_from_heatmap(batch_heatmaps, normalize=False):

    batch, height, width, n_points = tf.shape(batch_heatmaps)[0], tf.shape(
        batch_heatmaps)[1], tf.shape(batch_heatmaps)[2], tf.shape(batch_heatmaps)[3]

    batch_heatmaps = nms(batch_heatmaps)

    flat_tensor = tf.reshape(batch_heatmaps, (batch, -1, n_points))

    # Argmax of the flat tensor
    argmax = tf.argmax(flat_tensor, axis=1)
    argmax = tf.cast(argmax, tf.int32)
    scores = tf.math.reduce_max(flat_tensor, axis=1)

    # Convert indexes into 2D coordinates
    argmax_y = argmax // width
    argmax_x = argmax % width
    argmax_y = tf.cast(argmax_y, tf.float32)
    argmax_x = tf.cast(argmax_x, tf.float32)

    if normalize:
        argmax_x = argmax_x / tf.cast(width, tf.float32)
        argmax_y = argmax_y / tf.cast(height, tf.float32)

    # Shape: batch * 3 * n_points
    batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=1)
    # Shape: batch * n_points * 3
    batch_keypoints = tf.transpose(batch_keypoints, [0, 2, 1])

    return batch_keypoints
