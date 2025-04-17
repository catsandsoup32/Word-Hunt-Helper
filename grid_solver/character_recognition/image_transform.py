"""
For inference and dataset preprocessing
    - Crop white outer borders of image
    - Resize to 32x32
"""
from PIL import Image
import random
import cv2
import numpy as np

def PIL_transform(img: Image.Image, crop_percent=5, resize_len=32):
    width, height = img.size
    crop_pixels = (crop_percent/100 * width)*0.5 + (crop_percent/100 * height)*0.5
    crop_box = (crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels)
    cropped_img = img.crop(crop_box)
    resized_img = cropped_img.resize((resize_len, resize_len))
    return resized_img
