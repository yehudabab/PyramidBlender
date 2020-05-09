from blender import Blender
from improc_utils import *
from mask_generator import MaskGenerator

if __name__ == '__main__':
    pyr_blender = Blender(kernel_size=5)
    mask_gen = MaskGenerator()

    im1 = open_image('d:/lenna.png', rgb=False)
    im2 = open_image('d:/face.jpg', rgb=False)
    mask = mask_gen.split(0.5, 'h', shaped_like=im1)

    blended = pyr_blender.blend(im1, im2, mask)
    mixed = pyr_blender.mix(im1, im2, mask)

    display_many([blended, mixed])

