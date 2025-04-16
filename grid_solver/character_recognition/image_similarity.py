"""
Functions to compute difference between images 
"""
import numpy as np
import os
from PIL import Image
from scipy.fftpack import dct
import imagehash

clean_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_images")

def simple_pixel_diff(image):
    if isinstance(image, str): image = Image.open(image)
    image = np.asarray(image.resize((32, 32))) 
    min_diff = float("inf")
    min_idx = 0
    clean_images = os.listdir(clean_images_folder)
    for idx in range(len(clean_images)):
        clean_image = np.asarray(
            Image.open(f"{clean_images_folder}/{clean_images[idx]}").resize((32, 32)))
        diff = np.sum(np.abs(clean_image - image))
        if diff < min_diff:
            min_diff = diff
            min_idx = idx
    return chr(ord('a') + min_idx)

def p_hash(image):
    if isinstance(image, str): image = Image.open(image)
    return str(imagehash.phash(image))

def generate_ground_truth_hashes():
    hashes = []
    for file in os.listdir(clean_images_folder):
        hashes.append(p_hash(f"{clean_images_folder}/{file}"))
    return hashes

def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def p_hash_diff(image, clean_hashes):
    if isinstance(image, str): image = Image.open(image)
    min_diff = float("inf")
    min_idx = 0
    image_hash = p_hash(image)
    for idx, clean_hash in enumerate(clean_hashes):
        diff = hamming_distance(image_hash, clean_hash)
        if diff < min_diff:
            min_diff = diff
            min_idx = idx
    return chr(ord('a') + min_idx)

# clean_hashes = generate_ground_truth_hashes()
# print(p_hash_diff(
#     r"C:\Users\edmun\Desktop\VSCode Projects\Word-Hunt\grid_solver\character_recognition\s.PNG", clean_hashes))

