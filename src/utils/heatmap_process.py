from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np


def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[-1]):
        # ignore last channel, background channel
        _map = heatMap[:, :, i]
        _map = gaussian_filter(_map, sigma=0.5)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))
    return kplst


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))
