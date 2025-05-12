# HOG-LBP-Face-Recognition
This project implements a face recognition system using Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP) for feature extraction, combined with Support Vector Machine (SVM) for classification. The system includes preprocessing, feature extraction, dimensionality reduction (PCA), model training, and prediction.

## 📌 Key Features

Custom Dataset Handling: Loads and preprocesses grayscale face images from a directory structure.

Feature Extraction:

HOG: Captures edge and texture patterns.

LBP: Encodes local texture information.

Dimensionality Reduction: Uses PCA to reduce feature dimensions while preserving discriminative information.

SVM Classifier: Trains separate models for HOG and LBP features.

Prediction Pipeline: Processes new images and predicts identities using trained models.

## ⚙️ Workflow

#### 1. Data Loading & Preprocessing (Cell 1):

  Loads images from structured directories (each subfolder = one person).

  Applies histogram equalization and resizing for consistency.

  Splits data into train/test sets.

#### 2. Feature Extraction (Cell 2 & 3):

HOG: Extracts gradient-based features.

LBP: Computes local texture patterns.

Visualizes features for analysis.

#### 3. Feature Standardization & PCA (Cell 4):

Normalizes features using StandardScaler.

Reduces dimensions with PCA (whitening enabled).

#### 4. Model Training & Evaluation (Cell 5):

Trains SVM (RBF kernel) on HOG/LBP features.

Evaluates using classification reports and confusion matrices.

#### 5. Visualization & Prediction (Cell 6 & 7):

Displays sample faces and compares feature extraction times.

predict_person() function predicts identities for new images.

## 🚀 Usage
Place your face dataset in a directory where each subfolder is named after a person and contains their face images.

Adjust the DATASET_PATH to your dataset location.

Run the notebook or script to train models, evaluate them, and perform predictions.

Use the predict_person() function to recognize individuals from new images.
 ### 1. Setup
#### Clone the repo
```
git clone https://github.com/SehamElmaghraby/HOG-LBP-Face-Recognition.git
cd face-recognition-hog-lbp
```
#### Install dependencies
```
pip install numpy scikit-learn scikit-image matplotlib pillow
```  
Python 3.7+
scikit-learn, scikit-image, numpy, matplotlib, Pillow

### 2. Prepare Dataset
```
Store images in this structure:
dataset/
  ├── Person1/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── Person2/
  │   ├── image1.jpg
  │   └── image2.jpg
  ...
```
### 3. Run the Code
#### Update dataset path in Cell 1
```
DATASET_PATH = "path/to/your/dataset"
```
 Execute all cells in order:
1. Data Loading → 2. HOG → 3. LBP → 4. PCA → 5. Training → 6. Visualization → 7. Prediction

### 4. Predict on New Images
```
predicted_name = predict_person(
    "path/to/test_image.jpg",
    hog_clf,  # or lbp_clf
    pca_hog,  # or pca_lbp
    scaler_hog,  # or scaler_lbp
    feature_type="HOG"  # or "LBP"
)
print(f"Predicted: {predicted_name}")
```
## 📸 Sample
### 1. Input Image Example
![3](https://github.com/user-attachments/assets/b2cf5a82-a9fe-45f8-bc5e-59f49432022f)


### 2. Feature Extraction  

#### HOG Features
![HOG](https://github.com/user-attachments/assets/58ad13b1-97ee-4175-9993-f4701ec852a7)

*HOG highlights edges (eyes, nose, jawline)* 
#### LBP Features
![Local Binary Pattern](https://github.com/user-attachments/assets/1ea57d56-043b-4983-9027-9eba9a52ee1c)

*LBP encodes local texture patterns* 

## 🎯 Why This Project?
Demonstrates traditional ML (non-deep-learning) for face recognition.

Educational: Clear workflow for beginners in computer vision.
