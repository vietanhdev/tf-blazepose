import cv2


def visualize_keypoints(image, keypoints, visibility=None, edges=None, point_color=(0, 255, 0), text_color=(0, 0, 0)):
    """Visualize keypoints

    Args:
        image: Input image
        keypoints: Keypoints in format [[x1, y1], [x2, y2], ...]
        visibility [list]: List of visibilities of keypoints. 0: occluded, 1: visible

    Returns:
        Visualized image
    """

    draw = image.copy()
    for i, p in enumerate(keypoints):
        x, y = p[0], p[1]
        tmp_point_color = point_color
        if visibility is not None and not int(visibility[i]):
            tmp_point_color = (100, 100, 100)
        draw = cv2.circle(draw, center=(int(x), int(y)),
                          color=tmp_point_color, radius=5, thickness=-1)
        draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, text_color, 1, cv2.LINE_AA)

    if edges is not None and visibility is not None:
        for edge_chain in edges:
            for i in range(len(edge_chain) - 1):
                if visibility[edge_chain[i]] and visibility[edge_chain[i+1]]:
                    p1 = tuple(keypoints[edge_chain[i]])
                    p2 = tuple(keypoints[edge_chain[i+1]])
                    cv2.line(draw, p1, p2, (0, 0, 255), 2)

    return draw
