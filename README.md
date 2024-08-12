# Cancer Prediction Model

## Overview

This project involves building a machine learning model to predict whether a patient has cancer based on features extracted from breast cancer diagnostic data. We use the Breast Cancer Wisconsin dataset provided by `sklearn`, and implement a RandomForestClassifier to classify cancer as malignant or benign. The project includes data preprocessing, model training, evaluation, and prediction on new data.

## Features

- Data Exploration and Visualization
- Feature Engineering and Standardization
- Model Training with RandomForestClassifier
- Hyperparameter Tuning using GridSearchCV
- Model Evaluation using confusion matrix, classification report, and ROC curve
- Prediction on new data entries

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## Data

The dataset used is the Breast Cancer Wisconsin dataset, which is available through `sklearn.datasets`. The dataset includes various features related to breast cancer tumors and the target label indicating whether the tumor is malignant or benign.

## Usage

### 1. Load and Explore Data

The script loads the dataset and performs exploratory data analysis (EDA), including statistical summaries and visualizations.

### 2. Preprocess Data

Data preprocessing steps include feature scaling using `StandardScaler` and optional dimensionality reduction with PCA.

### 3. Train the Model

The model is trained using `RandomForestClassifier` with hyperparameter tuning through `GridSearchCV` to optimize performance.

### 4. Evaluate the Model

Model performance is evaluated using accuracy, confusion matrix, classification report, and ROC curve.

### 5. Predict New Data

You can use the trained model to predict whether a new data entry indicates a malignant or benign tumor. Follow the code in the script to enter new data and receive predictions.

## Example Code

Here's how you can use the trained model to make a prediction on new data:

```python
import numpy as np
import joblib

# Load the trained model and scaler
best_model = joblib.load('cancer_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# New data entry (example values)
new_data = np.array([[15.0, 15.0, 95.0, 600.0, 0.1, 0.2, 0.1, 0.2, 0.2, 0.06, 
                      0.4, 1.0, 3.0, 40.0, 0.005, 0.02, 0.02, 0.01, 0.02, 0.003,
                      18.0, 20.0, 120.0, 1000.0, 0.15, 0.35, 0.25, 0.4, 0.3, 0.08]])

# Preprocess the new data
new_data_scaled = scaler.transform(new_data)

# Predict the class and probability
prediction = best_model.predict(new_data_scaled)
prediction_proba = best_model.predict_proba(new_data_scaled)

# Output the result
cancer_type = 'Benign' if prediction[0] == 1 else 'Malignant'
print(f"The model predicts that the person has {cancer_type} cancer.")
print(f"Probability of being benign: {prediction_proba[0][1]:.2f}")
print(f"Probability of being malignant: {prediction_proba[0][0]:.2f}")
```

## Model Files

- `cancer_prediction_model.pkl`: The trained RandomForest model.
- `scaler.pkl`: The scaler used for feature standardization.
