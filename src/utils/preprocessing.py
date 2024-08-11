import cv2
import torch
import numpy as np

def remove_black_border(image):

    mask = image < 20
    mask = np.all(mask, axis=2)
    coords = np.argwhere(~mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    cropped = image[x0:x1, y0:y1]

    return cv2.resize(cropped, (image.shape[1], image.shape[0]))

def rgb2hsv(rgb: torch.Tensor) -> torch.Tensor:

    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
