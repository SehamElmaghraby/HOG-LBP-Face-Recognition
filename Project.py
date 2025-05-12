"""
Cell 1: 
Load custom face dataset and prepare train/test splits
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from skimage.filters import unsharp_mask
# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.25
RESIZED_HEIGHT = 150  # Matches LFW default resize factor of 0.5 (original is 125x94),not anymore
RESIZED_WIDTH = 112


"""Apply histogram equalization and resize to improve image resolution"""
def enhance_and_resize_image(image):
    # Enhance the image with histogram equalization
    image = exposure.equalize_hist(image)
    
    # Resize image to higher resolution
    image_resized = resize(image, (RESIZED_HEIGHT, RESIZED_WIDTH), anti_aliasing=True)
    
    return image_resized


"""Load and preprocess custom face dataset"""
def load_custom_dataset(dataset_path):
    print(f"Loading custom dataset from {dataset_path}...")
    
    # Get all subdirectories (each represents a person)
    person_names = [name for name in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, name))]
    
    images = []
    labels = []
    target_names = []
    
    for label, person_name in enumerate(person_names):
        person_dir = os.path.join(dataset_path, person_name)
        target_names.append(person_name)
        
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, image_file)
                
                try:
                    # Load and convert to grayscale
                    img = imread(image_path, as_gray=True)
                     # Apply enhancement and resize
                    img = enhance_and_resize_image(img)
                    
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    target_names = np.array(target_names)
    
    return X, y, target_names

def create_train_test_split(X, y):
    """Split dataset into training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    return X_train, X_test, y_train, y_test

# Load and split dataset (replace with your actual dataset path)
DATASET_PATH = r"path_to_dataset"
X, y, target_names = load_custom_dataset(DATASET_PATH)
X_train, X_test, y_train, y_test = create_train_test_split(X, y)

print(f"Dataset loaded with {len(target_names)} people")
print(f"Training set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")
