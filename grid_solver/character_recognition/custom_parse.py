# Helper to convert the grid photo into a grid of characters.

import cv2
import numpy as np
from PIL import Image

def parse_boxes(image_path_or_obj, show_process):
    """Returns a list of (x, y, w, h) tuples for bounding boxes in L2R T2B order, and the preprocessed image"""
    
    if isinstance(image_path_or_obj, str):
        image = cv2.imread(image_path_or_obj)
    elif isinstance(image_path_or_obj, np.ndarray):
        image = image_path_or_obj
    else:
        image = np.asarray(image_path_or_obj)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    width, height, _ = image.shape
    if show_process: output_image = image.copy() 

    min_area = (width / 5) * (height / 5) # Assumes that image is cropped just beyond the outer border
    max_area = (width * 0.9) * (height * 0.9)
    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
            if show_process:
                print(x, y, w, h)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    if show_process:
        print(f"Found {len(boxes)} boxes")
        cv2.imshow("Preprocessed", close)
        cv2.imshow("Orig with boxes", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sorted(boxes, key=lambda box: box[1] + box[0]*0.1), Image.fromarray(close)

def get_images(preprocessed_image, boxes, show_process):
    cells = []
    for box in boxes:
        # crop takes in (left, upper, right, lower)
        formatted_box = (box[0], box[1], box[0]+box[2], box[1]+box[3]) 
        cells.append(preprocessed_image.crop(formatted_box))
    if show_process: 
        for i in range(16): 
            break
            cells[i].show() 
            
    return cells # Grayscale PIL images

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# need character_recognition prepend when calling from grid_solver directory

from character_recognition.model import SmallCNN, logits_to_class
import torch, os
from character_recognition.image_transform import get_torch_transform, PIL_transform
from character_recognition.image_similarity import generate_ground_truth_hashes, p_hash_diff

model = SmallCNN()
model.load_state_dict(
    torch.load("character_recognition/weights/3_epochs.pth", 
               map_location=torch.device("cpu"),
               weights_only=True)
)
model.eval()
transform = get_torch_transform()

clean_hashes = generate_ground_truth_hashes()

def CNN_classify(image):
    transformed_image = transform(image).unsqueeze(dim=0)
    pred = model(transformed_image)
    pred = logits_to_class(pred)
    return pred

def get_grid(image_path_or_obj, show_process, grid):
    """Edits grid parameter to be nested list of characters.
       Uses perceptual hashing / CNN to convert image to character.
    """
    boxes, preprocessed_im = parse_boxes(image_path_or_obj, show_process)
    image_list = get_images(preprocessed_im, boxes, show_process)

    for i in range(4):
        row = []
        list_split = image_list[i*4: i*4 + 4]
        for image in list_split:
            image = PIL_transform(image)
            pred, confidence = p_hash_diff(image, clean_hashes)
            if (confidence < 1): # TODO: better hash function...
                pred = CNN_classify(image)
            row.append(pred)
        grid.append(row)
    

# Testing
if __name__ == "__main__":
    grid = []
    get_grid(
        "test2.jpg", 
        show_process=True, 
        grid=grid)
    print(grid)
