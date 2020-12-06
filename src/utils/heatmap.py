from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np


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
        [type]: [description]
    """
    npart = keypoints.shape[0]
    gtmap = np.zeros(shape=(heatmap_size[1], heatmap_size[0], npart), dtype=float)
    for i in range(npart):
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


def post_process_heatmap(heatmap):
    """Post process heatmap for keypoints

    Args:
        heatmap (np.array): Heatmap

    Returns:
        kplst: Keypoints in format [(x1, y1, confidence1), (x2, y2, confidence2), ...]
    """
    kplst = list()
    for i in range(heatmap.shape[-1]):
        hmap = heatmap[:, :, i]
        hmap = gaussian_filter(hmap, sigma=0.5)
        nms_peaks = non_max_supression(hmap, window_size=3, threshold=1e-6)

        y, x = np.where(nms_peaks == nms_peaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), nms_peaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))
    return kplst


def non_max_supression(plain, window_size=3, threshold=1e-6):
    """Non-max supression for heatmap"""
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))
