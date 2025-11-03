# Waste Management ML Predictor

## Project Overview
A machine learning application for predicting waste generation patterns, efficiency classification, and neighborhood clustering based on various demographic and environmental factors.

## Models
The project uses three machine learning models:

1. **Polynomial Regression**: Predicts waste generation per person (kg) based on neighborhood characteristics.
2. **Logistic Regression**: Classifies neighborhoods into efficiency categories (Very Efficient, Efficient, Moderate, Inefficient).
3. **KMeans Clustering**: Groups similar neighborhoods together for coordinated waste management strategies.

## Accuracy Metrics
The improved models achieve the following accuracy rates:
- **Polynomial Regression**: 99.43% (R²)
- **Logistic Regression**: 95.20% (Accuracy)
- **KMeans Clustering**: 16.78% (Silhouette Score)

## Installation & Setup

### Requirements
- Python 3.8+
- Flask
- scikit-learn
- numpy
- pandas
- matplotlib
- joblib

### Quick Start
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r smart_waste_ml/requirements.txt
   ```
3. Run the application:
   ```
   python run.py
   ```
4. Open your browser at `http://127.0.0.1:5000/`

## Project Structure
```
smart_waste_project/
├── fix_models.py            # Tool to fix model naming and format issues
├── improved_models.py       # Script to train high-accuracy ML models
├── rename_remaining_files.py # Tool to rename KNN files to KMeans
├── run.py                   # Main application entry point
├── train_models.py          # Original model training script
└── smart_waste_ml/          # Core package
    ├── data/                # Data storage
    ├── models/              # ML model definitions and saved models
    │   ├── metrics/         # Model performance metrics and visualizations
    │   ├── saved_models/    # Serialized model files
    │   ├── kmeans_clustering.py
    │   ├── logistic_regression.py
    │   └── polynomial_regression.py
    └── webapp/              # Flask web application
        ├── static/          # CSS, JS, and images
        ├── templates/       # HTML templates
        │   ├── index.html
        │   ├── prediction.html
        │   └── prediction_result.html
        └── app.py           # Flask application routes and logic
```

## Recent Fixes & Improvements

### 1. Model Compatibility
- Enhanced app.py to support both original and improved model formats
- Added helper functions for more robust model loading and prediction
- Improved error handling for model failures
- Renamed KNN Clustering to KMeans Clustering for accuracy and consistency

### 2. Interface Improvements  
- Fixed CSS styling issues in the prediction result template
- Added better error handling in templates for missing prediction results
- Streamlined the prediction form submission process

### 3. Code Quality
- Added fix_models.py script to ensure model format compatibility
- Reorganized model prediction code for better maintainability
- Added fallback mechanisms to ensure application works even with model errors

## Usage
1. Navigate to the "Make Predictions" page
2. Enter the neighborhood characteristics using the sliders and form fields
3. Click "Generate Predictions" to see the results from all three models

## Maintenance
If you encounter issues with the models:
1. Run `python fix_models.py` to fix model compatibility issues
2. Alternatively, run `python improved_models.py` to recreate all models

## License
Copyright © 2024 Waste Management ML Predictor 