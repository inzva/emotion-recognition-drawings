#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

import enum
from typing import Union, Tuple

from torchvision.transforms import transforms


class CropType(enum.Enum):
    NoCrop = 0
    Center = 1
    Random = 2

class ImageTransform:
    
    """ Parameters:
    * swap:                     sets how to swap the image dimensions, input image is in [h, w, c] 
                                and default swap which is [2, 0, 1]) converts it to [c, h, w].
    
    * grayscale:                the probability to convert image to a grayscale image. If image 
                                is converted to grayscale, then additional color distortion is 
                                not applied.
    
    * distort_prob:             probability to apply color distortion. Includes individual gains:
                                hue (distort_hgain), saturation (distort_sgain), distort_vgain
    
    * hflip_prob:               probability to horizontally flip the image
    
    * pad_to_square:            a boolean indicating if non-square images should be converted to
                                square images by padding with the pixel (114, 114, 114).
    
    * resize:                   gets an integer or a tuple of integers. If an integer is given, the
                                short side of the image is scaled to the given image and the h / w
                                ratio of the image is not broken. If a tuple is given as 
                                (width, height), then the image is directly scaled to the given size.
    
    * crop_type:                indicates the type of the crop. Includes NoCrop, Center, and Random.
                                The only supported version of cropping is square cropping. If cropping
                                is applied, then the resulting image h / w = 1.
    
    * crop_len_ratio:           gets a float, and indicates that the side length of the cropped image
                                is min(w, h) * crop_len_ratio.
    
    * crop_min_ratio:           only used in Random Cropping. While selecting the side length of the 
                                cropped image, the following is applied: random integer between 
                                (min(w, h) * crop_len_ratio * min_ratio)-1, min(w, h) * crop_len_ratio
                                It enables to further crop randomly smaller-sized images.
    
    * gaussian_noise_prob:      The probability of adding a gaussian noise to the image.
    
    * random_perspective_prob:  This module combines the rotation, shear, and scaling processes. It is
                                taken from the YOLOX repository and default parameters are set. This
                                parameter defines with how much probability this random perspective is
                                applied. This process further includes the following parameters: 
                                - degrees: rotation degree limit of the image
                                - translate
                                - scale
                                - shear
                                - perspective
                                - border
    """
    
    def __init__(self, 
                 swap: Tuple[int, int, int]=(2,0,1), 
                 grayscale_prob: float=0.0,
                 distort_prob: float=0.0,
                 distort_hgain: float=0.015, 
                 distort_sgain:float=0.7,
                 distort_vgain:float=0.4,
                 hflip_prob: float=0.0,
                 pad_to_square: bool=False,
                 resize_len: Union[int, Tuple[int, int], list]=-1, # -1 for no resize
                 crop_type: CropType=CropType.NoCrop,
                 crop_len_ratio: float=1.0,
                 crop_min_ratio: float=1.0,
                 gaussian_noise_prob: float=0.0,
                 gaussian_noise_pixel_prob: float=0.3,
                 random_perspective_prob: float=0.0,
                 degrees=10,
                 translate=0.1,
                 scale=0.9,
                 shear=10,
                 perspective=0.0,
                 border=(0, 0)
                ):
        
        self.swap = swap
        self.operations = [
            ColorDistortion(grayscale_prob, distort_prob, 
                            distort_hgain, distort_sgain, distort_vgain),
            HorizontalFlip(hflip_prob),
            GaussianNoise(gaussian_noise_prob, gaussian_noise_pixel_prob),
            Crop(crop_type, crop_len_ratio, crop_min_ratio),
            RandomPerspective(random_perspective_prob, degrees, translate, 
                              scale, shear, perspective, border),
        ]
        
        if pad_to_square:
            self.operations.append(PadSquare())
        
        if type(resize_len) != int or resize_len > 0.0:
            self.operations.append(Resize(resize_len))
            
        self.normalizer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((127.5,), (127.5,))
            ]
        )
    
    def __call__(self, image):
        
        for operation in self.operations:
            image = operation.__call__(image)
        
        # image = image.transpose(self.swap)
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = self.normalizer(image)
        return image


class ValTransform(ImageTransform):
    
    def __init__(self):
        super().__init__(
            grayscale_prob=0.0,
            distort_prob=0.0,
            hflip_prob=0.0,
            pad_to_square=True,
            resize_len=224,
            crop_type=CropType.NoCrop,
            gaussian_noise_prob=0.0, 
            random_perspective_prob=0.0)
    
class SlightTransform(ImageTransform):
    """ only includes: color distortion, horizontal flip, center crop """
    
    def __init__(self):
        super().__init__(
            grayscale_prob=0.0,
            distort_prob=0.5,
            hflip_prob=0.5,
            pad_to_square=False,
            resize_len=224,
            crop_type=CropType.Center,
            crop_len_ratio=1.0,
            gaussian_noise_prob=0.0, 
            random_perspective_prob=0.0)

class MediumTransform(ImageTransform):
    """ 
    only includes: grayscale, color distortion, 
    horizontal flip, center crop, gaussian noise 
    """
    
    def __init__(self):
        super().__init__(
            grayscale_prob=0.05,
            distort_prob=0.3,
            hflip_prob=0.5,
            pad_to_square=False,
            resize_len=224,
            crop_type=CropType.Center,
            crop_len_ratio=0.9,
            gaussian_noise_prob=0.25,
            gaussian_noise_pixel_prob=0.3,
            random_perspective_prob=0.0)
        
        
class FullTransform(ImageTransform):
    """ includes all augmentation methods """
    
    def __init__(self):
        super().__init__(
            grayscale_prob=0.05,
            distort_prob=0.3,
            hflip_prob=0.5,
            pad_to_square=False,
            resize_len=224,
            crop_type=CropType.Center,
            crop_len_ratio=0.9,
            gaussian_noise_prob=0.25,
            gaussian_noise_pixel_prob=0.3,
            random_perspective_prob=0.5)  
    
    
    
class ColorDistortion:
    
    def __init__(self, grayscale_prob: float=0.0, distort_prob: float=0.0, 
                 hgain: float=0.015, sgain: float=0.7, vgain: float=0.4):
        
        assert grayscale_prob <= 1.0 and grayscale_prob >= 0.0
        assert distort_prob <= 1.0 and distort_prob >= 0.0
        
        self.grayscale_prob = grayscale_prob
        self.distort_prob   = distort_prob
        self.hgain          = hgain
        self.sgain          = sgain
        self.vgain          = vgain
        
    def __call__(self, img):

        if random.random() < self.grayscale_prob:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            return gray_img
        
        elif random.random() < self.distort_prob:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8
        
            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
            img_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            ).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
            return img
        else:
            return img

class HorizontalFlip:
    
    def __init__(self, prob: float=0.0):
        assert prob >= 0.0 and prob <= 1.0
        self.p = prob
        
    def __call__(self, img):
        if random.random() < self.p:
            img = img[:, ::-1]
        return img


class Crop:
    
    def __init__(self, 
                 crop_type: CropType=CropType.NoCrop, 
                 len_ratio: float=1.0, 
                 min_ratio: float=1.0):
        assert len_ratio >= 0.0 and len_ratio <= 1.0
        assert min_ratio >= 0.0 and min_ratio <= 1.0
        
        self.crop_type = crop_type
        self.len_ratio = len_ratio
        self.min_ratio = min_ratio
        
    def __call__(self, image):
        h, w     = image.shape[:2]
        
        # print(self.crop_type.name, CropType.NoCrop.name, self.crop_type.name == CropType.NoCrop.name)
        
        if self.crop_type.name == CropType.NoCrop.name:
            return image
        
        elif self.crop_type.name == CropType.Center.name:
            cx, cy   = w // 2, h // 2
            min_len  = min_len  = int(min(h, w) * self.len_ratio / 2)
            image    = image[cy-min_len:cy+min_len, cx-min_len:cx+min_len, ...]
            return image
        
        elif self.crop_type.name == CropType.Random.name:
            min_len  = int(min(h, w) * self.len_ratio)
            side_len = np.random.randint(int(min_len * self.min_ratio)-1, min_len)
            x_start  = np.random.randint(0, w-side_len)
            y_start  = np.random.randint(0, h-side_len)
            
            image = image[y_start:y_start+side_len, x_start:x_start+side_len, ...]
            return image
        
        else:
            raise ValueError("Given Crop type is not implemented as augmentation technique!")
            
class PadSquare:
    
    def __init__(self, pad_pixel: float=114.0):
        assert pad_pixel >= 0.0 and pad_pixel <= 255.0
        self.pad_pixel = pad_pixel
    
    def __call__(self, image):
        h, w = image.shape[:2]
        
        if h == w:
            return image
        else:
            padded_img = np.ones((max(h, w), max(h, w), 3), dtype=np.uint8) * self.pad_pixel
            padded_img[:h, :w, ...] = image
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
            return padded_img

class Resize:
    
    def __init__(self, size):
        assert type(size) in [int, tuple, list]
        if type(size) != int:
            assert len(size) == 2
        else:
            assert size > 0
        
        self.size = size
    
    def __call__(self, image):
        """
        - image: cv2 image to be resized. Shape of [h, w, 3]
        - size : min_side_length or (width, height)
        
        If given a size as an Int, then the ratio of w / h is preserved.
        The smaller side is resized to the size. If given a tuple, then 
        the image is directly resized to the tuple dimension.
        """
        
        h, w = image.shape[:2]
        
        if type(self.size) == int:
        
            if h > w:
                size = (self.size, int(h * (self.size / w)))
            elif w > h:
                size = (int(w * (self.size / h)), self.size)
            else:
                size = (self.size, self.size)
        else:
            size = self.size
        
            
        resized_img = cv2.resize(image, size, 
                                 interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        return np.ascontiguousarray(resized_img, dtype=np.float32)
    
class GaussianNoise:
    
    def __init__(self, prob: float=0.3, pixel_prob: float=0.5):
        assert prob >= 0.0 and prob <= 1.0
        self.p = prob
        self.pixel_p = pixel_prob
    
    def __call__(self, img):
        if random.random() < self.p:
            gaussian_noise = np.random.rand(*img.shape[:-1], 1)
            img = img * (gaussian_noise < (1-self.pixel_p))
        return img

class RandomPerspective:
    
    def __init__(self, 
                 prob: float=0.5,
                 degrees: float=10,
                 translate: float=0.1,
                 scale: float=0.1,
                 shear: float=10,
                 perspective: float=0.0,
                 border: float=(0, 0)):
        
        assert prob >= 0.0 and prob <= 1.0
        
        self.p = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def __call__(self, img):
        
        if random.random() >= self.p:
            return img
        
        # targets = [cls, xyxy]
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2
    
        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
    
        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = self.scale # random.uniform(self.scale[0], self.scale[1])
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)
    
        # Translation
        T = np.eye(3)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        )  # x translation (pixels)
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * height
        )  # y translation (pixels)
    
        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    
        ###########################
        # For Aug out of Mosaic
        # s = 1.
        # M = np.eye(3)
        ###########################
    
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=(width, height), borderValue=(114, 114, 114)
                )
            else:  # affine
                img = cv2.warpAffine(
                    img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
                )
                
        return img

