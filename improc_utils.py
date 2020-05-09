import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt


def open_image(im_path, rgb=False):
    if rgb:
        return cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    else:
        im = cv2.imread(im_path, 0)
        return np.reshape(im, (im.shape[0], im.shape[1], 1))


def display_image(im):
    if im.shape[-1] == 1:
        plt.imshow(im.squeeze(), cmap='gray')
    else:
        plt.imshow(im.squeeze())
    plt.show()


def display_many(pyramid):
    h = pyramid[0].shape[0]
    w = np.sum([i.shape[1] for i in pyramid])
    c = pyramid[0].shape[2]

    im = np.zeros((h, w, c))

    offset = 0
    for p_im in pyramid:
        im_h = p_im.shape[0]
        im_w = p_im.shape[1]

        im[0:im_h, offset:(offset + im_w), :] = p_im
        offset += im_w
    display_image(im)


def get_gaussian_filter(k):
    flat_filter = np.zeros(k)

    for i in range(k):
        flat_filter[i] = scipy.misc.comb(k - 1, i, exact=True)
    flat_filter_coefficient = np.sum(flat_filter)

    kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            kernel[i, j] = flat_filter[i] * flat_filter[j]

    norm_coefficient = 1. / (flat_filter_coefficient ** 2)
    return norm_coefficient * kernel

