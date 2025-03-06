import cv2
import os
import numpy as np

def predict_defect(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: File does not exist at {image_path}")
    
    # Load the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Error: OpenCV could not read the image at {image_path}. Ensure it is a valid image file.")
    
    # Resize the image to the required dimensions
    img = cv2.resize(img, (128, 128))
    
    # Convert image to grayscale (optional, based on model requirements)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    img_gray = img_gray / 255.0
    
    # Reshape image for model input (if necessary)
    img_gray = np.expand_dims(img_gray, axis=0)  # Add batch dimension
    img_gray = np.expand_dims(img_gray, axis=-1)  # Add channel dimension
    
    return img_gray

# Define test image path
test_image = "C:/Users/MICK/Documents/Project/defect_detection_project/data/sample.jpg"

# Run prediction function and print success message
try:
    processed_image = predict_defect(test_image)
    print("✅ Image successfully processed for inference.")
except Exception as e:
    print(e)
