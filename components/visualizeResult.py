# visualizeResult.py
import cv2
import numpy as np

def visualizeResult(frame, depth_map, HEIGHT2, WIDTH2):
    image = cv2.resize(frame, (WIDTH2, HEIGHT2))
    pred_depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    img_out = np.hstack((image, pred_depth_map_colored))
    return pred_depth_map_colored