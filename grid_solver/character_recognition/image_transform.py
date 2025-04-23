from PIL import Image, ImageEnhance, ImageOps
import random
from torchvision import transforms
import numpy as np
import os
import cv2

def PIL_transform(img: Image.Image, crop_percent=5, resize_len=32):
    """Crop white outer borders and resize."""
    width, height = img.size
    crop_pixels = (crop_percent/100 * width)*0.5 + (crop_percent/100 * height)*0.5
    crop_box = (crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels)
    cropped_img = img.crop(crop_box)
    resized_img = cropped_img.resize((resize_len, resize_len))
    return resized_img

def transform_for_balancing(img: Image.Image):
    """Takes in already-processed images and makes small changes for more data."""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.90, 1.10))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.90, 1.10))
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return img

def get_mean_std():
    pixels = []
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for char_class in os.listdir(data_path):
        for file in os.listdir(f"{data_path}/{char_class}"):
            img = cv2.imread(f"{data_path}/{char_class}/{file}", cv2.IMREAD_GRAYSCALE)
            pixels.append(img.flatten())
    
    pixels = np.concatenate(pixels)
    mean = np.mean(pixels) / 255.0  
    std = np.std(pixels) / 255.0
    return [mean], [std]

def get_torch_transform():
    mean, std = get_mean_std()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

