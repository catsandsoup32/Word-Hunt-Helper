from openai import OpenAI
import os, string, io, base64, uuid, shutil, random
import cv2, numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image
from custom_parse import parse_boxes, get_images 
from image_transform import PIL_transform, transform_for_balancing

def create_subfolders():
    parent_folder = "data"
    os.makedirs(parent_folder, exist_ok=True)
    for letter in string.ascii_lowercase:
        subfolder_path = os.path.join(parent_folder, letter)
        os.makedirs(subfolder_path, exist_ok=True)

def classify_gpt(PIL_image):
    buffered = io.BytesIO()
    PIL_image.save(buffered, format="JPEG")  # Use "PNG" or another format if needed
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    prompt = """Return the character in this image as a single lowercase character. Output only the character."""
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4.1-mini",
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

def classify_entire_grid(image_path_or_obj, specify_classes=None):
    """Split image into 16 and classify each with ChatGPT API, then save to folder."""
    boxes, preprocessed_im = parse_boxes(image_path_or_obj, False)
    image_list = get_images(preprocessed_im, boxes, False)

    # PIL image
    for image in image_list:
        #char_class = classify_gpt(image)
        class_folder = os.path.join(f"data", "q")
        image = PIL_transform(image) # Crop white edges and resize
        if os.path.exists(class_folder):
            image.save(os.path.join(class_folder, f"{uuid.uuid4()}.png"))
        
    #shutil.move(image_path_or_obj, f"pre_data/used")

def remove_all_data():
    for char_class in os.listdir("data"):
        for file in os.listdir(f"data/{char_class}"):    
            os.remove(f"data/{char_class}/{file}")

def move_data_back():
    for file in os.listdir("pre_data/used"):
        shutil.move(f"pre_data/used/{file}", "pre_data/unused")

def balance_dataset():
    """Balance dataset such that each class has an equal number of images."""
    class_to_len = {}
    max_len = 0
    for char_class in os.listdir("data"):
        curr_len = len(os.listdir(f"data/{char_class}"))
        class_to_len[char_class] = curr_len
        if curr_len > max_len: max_len = curr_len
    
    for char_class in os.listdir("data"):
        class_len = class_to_len[char_class]
        class_dir = os.listdir(f"data/{char_class}")
        for i in range(max_len - class_len):
            rand_sample_idx = random.randint(0, class_len-1)
            img = Image.open(f"data/{char_class}/{class_dir[rand_sample_idx]}")
            img = transform_for_balancing(img)
            img.save(f"data/{char_class}/SYNTHETIC{i}.png")

def remove_balancing():
    for char_class in os.listdir("data"):
        for file in os.listdir(f"data/{char_class}"):
            if (file.startswith("SYNTHETIC")):
                os.remove(f"data/{char_class}/{file}")

def crop_and_resize_clean_images():
    for file in os.listdir("clean_images"):
        img = Image.open(f"clean_images/{file}")
        img = PIL_transform(img)
        img.save(f"clean_images/{file}")

if __name__ == "__main__":
    # load_dotenv()
    # files = os.listdir("pre_data/unused")
    # for file in tqdm(files, "Labeling progess..."):
    #     classify_entire_grid(f"pre_data/unused/{file}")
    # remove_all_data()
    # move_data_back()
    # remove_balancing()
    # balance_dataset()
    crop_and_resize_clean_images()