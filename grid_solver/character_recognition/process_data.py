"""
Classify photos of Word Hunt grid, saving as data in torch ImageFolder format.
Goal: 
~50 laptop / Pi camera photos (low quality and realistic)
~50 phone screenshots with artifical degrading
"""

from openai import OpenAI
import os
import string
import io
import base64
import uuid
from dotenv import load_dotenv
from PIL import Image
from custom_parse import parse_boxes, get_images
from image_transform import PIL_transform

def create_subfolders():
    parent_folder = r"C:\Users\edmun\Desktop\VSCode Projects\Word-Hunt\grid_solver\character_recognition\data"
    # Make sure the parent folder exists
    os.makedirs(parent_folder, exist_ok=True)

    # Loop through letters a to z
    for letter in string.ascii_lowercase:
        subfolder_path = os.path.join(parent_folder, letter)
        os.makedirs(subfolder_path, exist_ok=True)
    print("Subfolders a to z created inside 'data'.")


def classify_gpt(PIL_image):
    buffered = io.BytesIO()
    PIL_image.save(buffered, format="JPEG")  # Use "PNG" or another format if needed
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    prompt = "Return the character in this image as a single lowercase character. Output only the character."
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    },
                ],
            }
        ],
    )
    return response.output_text

# Split image into 16 and classify each with ChatGPT API, then save to folder
def classify_entire_grid(image_path_or_obj):
    boxes, preprocessed_im = parse_boxes(image_path_or_obj, False)
    image_list = get_images(preprocessed_im, boxes, False)

    # PIL image
    for image in image_list:
        char_class = classify_gpt(image)
        class_folder = os.path.join(f"data", char_class)
        image = PIL_transform(image)
        image.save(os.path.join(class_folder, f"{uuid.uuid4()}.png"))
        break
        

load_dotenv()
classify_entire_grid(Image.open(r"C:\Users\edmun\Desktop\VSCode Projects\Word-Hunt\grid_solver\character_recognition\pre_data\used\4.jpg"))