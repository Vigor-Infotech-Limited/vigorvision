import random
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as F


class SegCompose:
    """
    Composes multiple segmentation transforms into one.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """
    Randomly horizontally flips the image and mask with a given probability.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    """
    Randomly vertically flips the image and mask with a given probability.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class RandomRotation:
    """
    Random rotation of image and mask (in degrees).
    """

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle, resample=Image.BILINEAR)
        mask = F.rotate(mask, angle, resample=Image.NEAREST)
        return image, mask


class Resize:
    """
    Resize image and mask to a fixed size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return image, mask


class RandomCrop:
    """
    Randomly crop image and mask to a given size.
    """

    def __init__(self, size):
        self.size = size  # (width, height)

    def __call__(self, image, mask):
        w, h = image.size
        th, tw = self.size
        if w == tw and h == th:
            return image, mask

        if w < tw or h < th:
            image = ImageOps.pad(image, self.size)
            mask = ImageOps.pad(mask, self.size, method=Image.NEAREST)
            return image, mask

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        image = image.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        return image, mask


class ToTensor:
    """
    Converts PIL Image and mask to Tensor. Mask is long (int64) for CE loss.
    """

    def __call__(self, image, mask):
        return F.to_tensor(image), torch.from_numpy(np.array(mask, dtype=np.int64))


class Normalize:
    """
    Normalize image tensor using mean and std. Mask remains unchanged.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, self.mean, self.std)
        return image, mask
    
class RandomBrightnessContrast:
    """
    Randomly adjust brightness and contrast of the input image.
    """

    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, mask):
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)

        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        return image, mask
    
import cv2

class CLAHEEqualization:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image, mask):
        np_img = np.array(image)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            for i in range(3):
                np_img[:, :, i] = self.clahe.apply(np_img[:, :, i])
        else:
            np_img = self.clahe.apply(np_img)
        image = Image.fromarray(np_img)
        return image, mask

from scipy.ndimage import gaussian_filter, map_coordinates

class ElasticTransform:
    """
    Applies elastic deformation to the image and mask.
    """

    def __init__(self, alpha=50, sigma=5):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image, mask):
        image_np = np.array(image)
        mask_np = np.array(mask)

        shape = image_np.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        def warp_array(arr, order=1):
            if arr.ndim == 3:
                result = np.zeros_like(arr)
                for i in range(arr.shape[2]):
                    result[..., i] = map_coordinates(arr[..., i], indices, order=order, mode='reflect').reshape(shape)
                return result
            else:
                return map_coordinates(arr, indices, order=order, mode='reflect').reshape(shape)

        warped_image = warp_array(image_np, order=1)
        warped_mask = warp_array(mask_np, order=0)

        image = Image.fromarray(np.uint8(warped_image))
        mask = Image.fromarray(np.uint8(warped_mask))
        return image, mask
