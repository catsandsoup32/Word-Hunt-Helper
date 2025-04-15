"""
Functions to compute difference between images 
"""
import numpy as np

# # Test recognizer 
# test_image = Image.open(
#     r"C:\Users\edmun\Desktop\VSCode Projects\Word-Hunt\grid_solver\character_recognition\data\pre_data_used\ef939dbe-3b82-48b6-8101-d0954b8e906a.png")

# minDiff = 10000000000000000
# for f in os.listdir(f"{DATA_PATH}/train"):
#     for c in os.listdir(f"{DATA_PATH}/train/{f}"):
#         compareIm = Image.open(f"{DATA_PATH}/train/{f}/{c}")
#         arr1 = np.array(test_image)
#         arr2 = np.array(compareIm)
#         diff = np.abs(arr1.astype(int) - arr2.astype(int))
#         diff = np.sum(diff)

#         if (diff < minDiff):
#             minDiff = diff
#             minPath = f"{f}"

# print(minPath)

def simple_pixel_diff(image):
    pass

def d_hash(image):
    pass
    

def generate_ground_truth_hashes(clean_images_path):
    pass

# Discrete cosine transform  