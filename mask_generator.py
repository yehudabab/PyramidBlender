import numpy as np


class MaskGenerator:
    def __init__(self):
        pass

    @staticmethod
    def circle(x, y, radius, shaped_like):
        mask = np.zeros_like(shaped_like)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - x) ** 2 + (j - y) ** 2 < radius ** 2:
                    mask[i, j, :] = 1
        return mask

    @staticmethod
    def split(ratio, axis, shaped_like):
        mask = np.zeros_like(shaped_like)
        h = mask.shape[0]
        v = mask.shape[1]

        length = h if axis == 'h' else v

        for i in range(h):
            for j in range(v):
                variable = i if axis == 'h' else j
                if variable < ratio * length:
                    mask[i, j, :] = 1
        return mask

    @staticmethod
    def cube(top, left, height, width, shaped_like):
        mask = np.zeros_like(shaped_like)
        h = mask.shape[0]
        v = mask.shape[1]

        for i in range(h):
            for j in range(v):
                if top < i < top + height and left < j < left + width:
                    mask[i, j, :] = 1
        return mask
