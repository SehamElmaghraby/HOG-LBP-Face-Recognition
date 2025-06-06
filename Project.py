"""
Cell 1 
Load custom face dataset and prepare train/test splits
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.25
RESIZED_HEIGHT = 150  
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

"""
Cell 2
Implement HOG feature extraction and visualization
"""

from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

def extract_hog_features(images):
    """Extract HOG features from images"""
    hog_features = []
    for image in images:
        fd = hog(image, 
                 orientations=HOG_ORIENTATIONS,
                 pixels_per_cell=HOG_PIXELS_PER_CELL,
                 cells_per_block=HOG_CELLS_PER_BLOCK,
                 visualize=False, 
                 channel_axis=None)
        hog_features.append(fd)
    return np.array(hog_features)

def visualize_hog_features(image):
    """Visualize HOG features for a sample image"""
    fd, hog_image = hog(image, 
                        orientations=HOG_ORIENTATIONS,
                        pixels_per_cell=HOG_PIXELS_PER_CELL,
                        cells_per_block=HOG_CELLS_PER_BLOCK,
                        visualize=True, 
                        channel_axis=None)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 20))
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG Features')
    ax2.axis('off')
    
    plt.suptitle('Histogram of Oriented Gradients (HOG)')
    plt.show()

# Extract HOG features
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Visualize for first training image
visualize_hog_features(X_train[15])

print(f"HOG feature shape: {X_train_hog.shape[1]} dimensions per image")



"""
Cell 3
Implement LBP feature extraction and visualization
"""

from skimage.feature import local_binary_pattern

# LBP parameters
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

def extract_lbp_features(images):
    """Extract LBP features from images"""
    lbp_features = []
    for image in images:
        
        lbp = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)
        hist, _ = np.histogram(lbp.ravel(), 
                              bins=np.arange(0, N_POINTS + 3),
                              range=(0, N_POINTS + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalize
        lbp_features.append(hist)
    return np.array(lbp_features)

def visualize_lbp_features(image):
    """Visualize LBP features for a sample image"""
    lbp = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    ax2.imshow(lbp, cmap=plt.cm.gray)
    ax2.set_title('LBP Features')
    ax2.axis('off')
    
    plt.suptitle('Local Binary Patterns (LBP)')
    plt.show()

# Extract LBP features
X_train_lbp = extract_lbp_features(X_train)
X_test_lbp = extract_lbp_features(X_test)

# Visualize for first training image
visualize_lbp_features(X_train[15])

print(f"LBP feature shape: {X_train_lbp.shape[1]} dimensions per image")



"""
Cell 4 
feature preprocessing pipeline
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PCA parameters
N_COMPONENTS = 150

def preprocess_features(X_train, X_test):
    """Standardize features and apply PCA"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=min(N_COMPONENTS, X_train_scaled.shape[1]), 
              whiten=True, 
              random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, X_test_pca, pca

# Preprocess HOG features
X_train_hog_pca, X_test_hog_pca, pca_hog = preprocess_features(X_train_hog, X_test_hog)

# Preprocess LBP features
X_train_lbp_pca, X_test_lbp_pca, pca_lbp = preprocess_features(X_train_lbp, X_test_lbp)

print(f"After PCA - HOG features: {X_train_hog_pca.shape[1]} dimensions")
print(f"After PCA - LBP features: {X_train_lbp_pca.shape[1]} dimensions")


"""
Cell 5
Implement classification and evaluation
"""

from sklearn.svm import SVC
from sklearn.metrics import classification_report 


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_type):
    """Train SVM classifier and evaluate performance"""
    print(f"\nEvaluating {feature_type} features...")
    
    # Train SVM
    clf = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names,zero_division=0))
    
    # Return the trained classifier (this was missing)
    return clf

# Create global references to scalers and PCAs
scaler_hog = StandardScaler()
scaler_lbp = StandardScaler()

# Standardize features before PCA
X_train_hog_scaled = scaler_hog.fit_transform(X_train_hog)
X_test_hog_scaled = scaler_hog.transform(X_test_hog)

X_train_lbp_scaled = scaler_lbp.fit_transform(X_train_lbp)
X_test_lbp_scaled = scaler_lbp.transform(X_test_lbp)

# Apply PCA
pca_hog = PCA(n_components=min(N_COMPONENTS, X_train_hog_scaled.shape[1]), 
          whiten=True, random_state=RANDOM_STATE)
X_train_hog_pca = pca_hog.fit_transform(X_train_hog_scaled)
X_test_hog_pca = pca_hog.transform(X_test_hog_scaled)

pca_lbp = PCA(n_components=min(N_COMPONENTS, X_train_lbp_scaled.shape[1]), 
          whiten=True, random_state=RANDOM_STATE)
X_train_lbp_pca = pca_lbp.fit_transform(X_train_lbp_scaled)
X_test_lbp_pca = pca_lbp.transform(X_test_lbp_scaled)

print(f"After PCA - HOG features: {X_train_hog_pca.shape[1]} dimensions")
print(f"After PCA - LBP features: {X_train_lbp_pca.shape[1]} dimensions")

# Evaluate HOG features and save classifier (fix the missing assignment)
hog_clf = train_and_evaluate(X_train_hog_pca, X_test_hog_pca, y_train, y_test, "HOG")

# Evaluate LBP features and save classifier (fix the missing assignment)
lbp_clf = train_and_evaluate(X_train_lbp_pca, X_test_lbp_pca, y_train, y_test, "LBP")

"""
Cell 6
Create visualizations and final report
"""

"""Display sample faces from dataset"""
def plot_sample_faces(images, labels, title):
    plt.figure(figsize=(12, 8))
    for i in range(min(12, len(images))):  # Handle case with fewer than 12 images
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(target_names[labels[i]], size=10)
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(title, size=16)
    plt.tight_layout()
    plt.show()

"""Compare computation times for feature extraction"""
def compare_feature_times():
    import time
    
    # Time HOG extraction
    start = time.time()
    _ = extract_hog_features(X_train[:10])
    hog_time = time.time() - start
    
    # Time LBP extraction
    start = time.time()
    _ = extract_lbp_features(X_train[:10])
    lbp_time = time.time() - start
    
    print(f"\nFeature Extraction Times (for 10 images):")
    print(f"HOG: {hog_time:.4f} seconds")
    print(f"LBP: {lbp_time:.4f} seconds")

# Display sample faces
plot_sample_faces(X_train, y_train, "Sample Faces from Data Set train")

# Compare feature extraction times
compare_feature_times()

"""
cell 7
Prediction function and test
"""

"""Predict person's name from an image file"""
def predict_person(image_path, classifier, pca, scaler, feature_type="HOG"):
    try:
        # Load and preprocess image
        img = imread(image_path, as_gray=True)
        
        # Apply the same enhancement used during training
        img = enhance_and_resize_image(img)
        
        # Extract features based on feature_type
        if feature_type == "HOG":
            features = extract_hog_features([img])
        elif feature_type == "LBP":
            features = extract_lbp_features([img])
        else:
            raise ValueError("feature_type must be either 'HOG' or 'LBP'")
        
        # Apply preprocessing (standardization and PCA)
        features_scaled = scaler.transform(features)  # Now uses the scaler parameter
        features_pca = pca.transform(features_scaled)  # Use the trained PCA
        
        # Predict
        pred_label = classifier.predict(features_pca)[0]
        return target_names[pred_label]

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Example usage with the pre-trained classifier, scaler, and PCA
test_image_path = r"path\to\test\image.jpg"
# Now correctly passing all required parameters
predicted_name_hog = predict_person(
    test_image_path, 
    hog_clf, 
    pca_hog,
    scaler_hog,
    "HOG"
)
print(f"Predicted person (HOG): {predicted_name_hog}")

