import cv2
import numpy as np
from PIL import Image

def parse_boxes(image_path: str, show_process: bool):
    """Returns a list of (x, y, w, h) tuples for bounding boxes in L2R T2B order"""
    image = cv2.imread(image_path)
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

    return sorted(boxes, key=lambda box: box[1] + box[0]*0.1)

def get_grid(image_path: str, show_process: bool):
    """Returns grid as nested list"""
    boxes = parse_boxes()
    # Need to get screenshots now


test = parse_boxes("grid_solver/character_recognition/JsxLT.jpg", False)