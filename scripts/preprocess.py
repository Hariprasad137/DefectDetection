import os
import cv2
import numpy as np

DATASET_PATH = "C:/Users/MICK/Documents/Project/defect_detection_project/data/piston_images/"

OUTPUT_PATH = "../data/processed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize
    return img

def preprocess_dataset():
    for category in ["good", "defective"]:
        class_dir = os.path.join(DATASET_PATH, category)
        output_dir = os.path.join(OUTPUT_PATH, category)
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            processed_img = preprocess_image(img_path)
            np.save(os.path.join(output_dir, img_name.split('.')[0] + ".npy"), processed_img)

preprocess_dataset()
print("Preprocessing complete!")
