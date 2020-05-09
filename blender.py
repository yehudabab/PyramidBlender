import numpy as np
from scipy import ndimage
from improc_utils import get_gaussian_filter, display_image
import math


class Blender:
    def __init__(self, kernel_size, debug=False):
        self.gaussian_kernel = get_gaussian_filter(kernel_size)
        self.debug = debug

    def blend(self, im1, im2, mask):
        assert im1.shape == im2.shape
        assert im1.shape == mask.shape

        num_reductions = int(math.log2(im1.shape[0]))

        l_pyr1 = self.__create_laplacian(im1, num_reductions)
        l_pyr2 = self.__create_laplacian(im2, num_reductions)

        mask_g_pyr = self.__create_gaussian(mask, num_reductions)

        return self.__apply_mask(l_pyr1, l_pyr2, mask_g_pyr)

    @staticmethod
    def mix(im1, im2, mask):
        assert im1.shape == im2.shape
        assert im1.shape == mask.shape
        return mask * im1 + (1 - mask) * im2

    def __check_laplacian(self, orig_im, pyramid):
        reconstructed = self.__sum_laplacian_pyramid(pyramid)
        assert np.all(orig_im == reconstructed)

    def __create_laplacian(self, im, num_reductions):
        gauss = self.__create_gaussian(im, num_reductions=num_reductions)
        pyr = self.__create_laplacian_from_gaussian(gauss)
        self.__check_laplacian(im, pyr)
        return pyr

    def __create_laplacian_from_gaussian(self, gaussian_pyramid):
        laplacian = [gaussian_pyramid[-1]]
        for g_i in range(len(gaussian_pyramid) - 1)[::-1]:
            expanded = self.__expand_image(gaussian_pyramid[g_i + 1])
            laplacian.append(gaussian_pyramid[g_i] - expanded)
        return laplacian[::-1]

    def __create_gaussian(self, im, num_reductions):
        pyramid = [im]
        for _ in range(num_reductions):
            reduced = self.__reduce_image(pyramid[-1])
            pyramid.append(reduced)
        return pyramid

    def __apply_mask(self, l_pyr1, l_pyr2, mask_g_pyr):
        combined_pyr = []

        for k in range(len(l_pyr1)):
            combined_pyr.append(mask_g_pyr[k] * l_pyr1[k] + (1 - mask_g_pyr[k]) * l_pyr2[k])

        recon = self.__sum_laplacian_pyramid(combined_pyr)
        recon[recon < 0] = 0
        recon[recon > 255] = 255
        return recon

    def __sum_laplacian_pyramid(self, pyr):
        sum_im = np.zeros_like(pyr[0])
        for i in range(len(pyr)):
            sum_im += self.__expand_image_n(pyr[i], i)
        return sum_im.astype(np.uint8)

    def __reduce_image(self, im):
        blurred_image = self.__blur(im, ker_factor=1.0)
        return blurred_image[::2, ::2, :]

    def __expand_image(self, im):
        padded_im = np.zeros((2 * im.shape[0], 2 * im.shape[1], im.shape[2]))
        padded_im[::2, ::2, :] = im
        result_im = self.__blur(padded_im, ker_factor=4.0)
        return result_im

    def __expand_image_n(self, im, n):
        res = im.copy()
        for _ in range(n):
            res = self.__expand_image(res)
        return res

    def __blur(self, im, ker_factor=1.0):
        kernel = ker_factor * self.gaussian_kernel
        result = np.zeros_like(im)
        for i in range(im.shape[-1]):
            result[:, :, i] = ndimage.convolve(im[:, :, i], kernel, mode='wrap')
        return result
