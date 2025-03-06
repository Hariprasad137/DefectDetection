import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

# Define dataset path (Update this after adding Kaggle dataset)
DATASET_PATH = "C:/Users/MICK/Documents/Project/defect_detection_project/data/piston_images/"


# Load dataset (Placeholder: Modify for your dataset)
def load_data():
    images, labels = [], []
    for category in ["good", "defective"]:  # Assuming two classes
        class_dir = os.path.join(DATASET_PATH, category)
        label = 0 if category == "good" else 1
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return np.array(images) / 255.0, np.array(labels)

# Prepare data
X, y = load_data()
X = X.astype("float32")
y = tf.keras.utils.to_categorical(y, 2)

# Define model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two classes: good & defective
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# Save model
model.save('defect_model.h5')

print("✅ Model saved successfully as defect_model.h5")

