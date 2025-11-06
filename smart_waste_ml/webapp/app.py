from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pandas as pd
import numpy as np
import os
import sys
import joblib
import traceback

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.secret_key = 'waste_management_predictor_key'  # Required for flash messages

# Paths to saved models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')
POLY_REG_MODEL_PATH = os.path.join(MODEL_DIR, 'polynomial_regression_model.joblib')
LOG_REG_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_clustering_model.joblib')

# Try to use improved models if they exist
IMPROVED_POLY_PATH = os.path.join(MODEL_DIR, 'improved_polynomial_regression_model.joblib')
IMPROVED_LOG_PATH = os.path.join(MODEL_DIR, 'improved_logistic_regression_model.joblib')
IMPROVED_KMEANS_PATH = os.path.join(MODEL_DIR, 'improved_kmeans_clustering_model.joblib')

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used by improved models when base inputs are present.
    This function is safe to call regardless of whether the model uses them.
    """
    df = df.copy()
    # Safely compute engineered fields if base fields exist
    if all(col in df.columns for col in ['commercial_activity', 'population_density']):
        df['commercial_density_ratio'] = df['commercial_activity'] * df['population_density'] / 1000.0
    if all(col in df.columns for col in ['recycling_rate', 'public_awareness']):
        df['recycling_awareness'] = df['recycling_rate'] * df['public_awareness']
    if 'month' in df.columns:
        # Seasonal factor similar to improved data generation
        df['seasonal_factor'] = np.sin(df['month'] * np.pi / 6.0) * 2.0
    if all(col in df.columns for col in ['is_weekend', 'is_holiday']):
        df['weekend_holiday'] = ((df['is_weekend'] + df['is_holiday']) > 0).astype(int)
    if 'weather_temperature' in df.columns:
        df['temp_squared'] = df['weather_temperature'] ** 2
    return df

def _prepare_features_for_model(model_blob, df: pd.DataFrame):
    """Given a loaded model blob (dict or estimator), select and scale features as expected by the model.
    Returns (X_ready, estimator) where X_ready is numpy array or DataFrame matching estimator expectations.
    """
    # model_blob could be:
    # - a dict with keys like 'model', 'feature_names', 'scaler'
    # - a sklearn Pipeline/estimator directly
    if isinstance(model_blob, dict):
        estimator = model_blob.get('model', None)
        feature_names = model_blob.get('feature_names', None)
        scaler = model_blob.get('scaler', None)
        # Compute engineered features so required columns exist
        enriched = _add_engineered_features(df)
        if feature_names is not None:
            missing = [c for c in feature_names if c not in enriched.columns]
            if missing:
                print(f"Warning: Missing expected features for model: {missing}")
            X = enriched.reindex(columns=feature_names, fill_value=0)
        else:
            X = enriched
        if scaler is not None:
            try:
                X_ready = scaler.transform(X)
            except Exception:
                # If scaler expects DataFrame vs ndarray, adjust accordingly
                X_ready = scaler.transform(np.asarray(X))
        else:
            X_ready = X
        # Fallback if 'model' key missing and blob itself is estimator
        if estimator is None:
            estimator = model_blob
        return X_ready, estimator
    else:
        # Estimator or Pipeline directly. Try to respect common app selection but still enrich features
        enriched = _add_engineered_features(df)
        return enriched, model_blob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET'])
def prediction_form():
    return render_template('prediction.html')

@app.route('/prediction', methods=['POST'])
def prediction_process():
    try:
        # Print debug information
        print("Form submitted, processing prediction...")
        print(f"Form data: {request.form}")
        
        # Extract form data
        input_data = {
            'population_density': float(request.form.get('population_density', 1000)),
            'income_level': float(request.form.get('income_level', 50000)),
            'recycling_rate': float(request.form.get('recycling_rate', 0.3)),
            'public_awareness': float(request.form.get('public_awareness', 5)),
            'commercial_activity': float(request.form.get('commercial_activity', 5)),
            'weather_temperature': float(request.form.get('weather_temperature', 20)),
            'is_holiday': 1 if request.form.get('is_holiday') == 'on' else 0,
            'is_weekend': 1 if request.form.get('is_weekend') == 'on' else 0,
            'month': int(request.form.get('month', 6)),
            'day_of_week': int(request.form.get('day_of_week', 3))
        }
        
        print(f"Processed input data: {input_data}")
        
        # Prepare features for prediction
        features = pd.DataFrame([input_data])
        print(f"Feature dataframe created: {features.shape}")
        
        # 1. Polynomial Regression model
        waste_prediction = predict_waste_generation(features)
        print(f"Waste prediction result: {waste_prediction}")
        
        # 2. Logistic Regression (efficiency classification)
        efficiency_class, efficiency_probabilities = predict_efficiency(features)
        print(f"Efficiency class prediction: {efficiency_class}")
        
        # 3. KMeans Clustering
        cluster_assignment = predict_cluster(features)
        print(f"Cluster assignment: {cluster_assignment}")
        
        # Find nearest neighbors in the same cluster (simulated)
        neighbor_count = 5
        neighbors = np.random.randint(1, 100, size=neighbor_count).tolist()
        
        # Prepare results
        result = {
            'waste_prediction': round(waste_prediction, 2),
            'efficiency_class': int(efficiency_class),
            'efficiency_probabilities': efficiency_probabilities.tolist(),
            'cluster_assignment': int(cluster_assignment),
            'neighbors': neighbors
        }
        
        print(f"Final results: {result}")
        print("Rendering prediction_result.html template...")
        
        return render_template('prediction_result.html', result=result, input_data=input_data)
    
    except Exception as e:
        print(f"Error in prediction process: {e}")
        print(traceback.format_exc())
        flash(f"An error occurred during prediction: {str(e)}", 'error')
        return render_template('prediction.html', error=str(e))

def predict_waste_generation(features):
    """Predict waste generation using either improved or original polynomial regression model"""
    try:
        # Try improved model first
        if os.path.exists(IMPROVED_POLY_PATH):
            print(f"Using improved polynomial regression model")
            model_blob = joblib.load(IMPROVED_POLY_PATH)
            X_ready, estimator = _prepare_features_for_model(model_blob, features)
            waste_prediction = estimator.predict(X_ready)[0]
            return waste_prediction
        
        # Fallback to original model
        print(f"Using original polynomial regression model")
        model_blob = joblib.load(POLY_REG_MODEL_PATH)
        X_ready, estimator = _prepare_features_for_model(model_blob, features)
        waste_prediction = estimator.predict(X_ready)[0]
        return waste_prediction
        
    except Exception as e:
        print(f"Error in waste prediction: {e}")
        print(traceback.format_exc())
        # Return a default prediction if model fails
        return 1.5

def predict_efficiency(features):
    """Predict efficiency class using either improved or original logistic regression model"""
    try:
        # Try improved model first
        if os.path.exists(IMPROVED_LOG_PATH):
            print(f"Using improved logistic regression model")
            model_blob = joblib.load(IMPROVED_LOG_PATH)
            X_ready, estimator = _prepare_features_for_model(model_blob, features)
            efficiency_class = estimator.predict(X_ready)[0]
            efficiency_probabilities = estimator.predict_proba(X_ready)[0]
            return efficiency_class, efficiency_probabilities
        
        # Fallback to original model
        print(f"Using original logistic regression model")
        model_blob = joblib.load(LOG_REG_MODEL_PATH)
        X_ready, estimator = _prepare_features_for_model(model_blob, features)
        efficiency_class = estimator.predict(X_ready)[0]
        efficiency_probabilities = estimator.predict_proba(X_ready)[0]
        return efficiency_class, efficiency_probabilities
        
    except Exception as e:
        print(f"Error in efficiency prediction: {e}")
        print(traceback.format_exc())
        # Return default values if model fails
        return 1, np.array([0.1, 0.7, 0.1, 0.1])

def predict_cluster(features):
    """Predict cluster using either improved or original KMeans model"""
    try:
        # Try improved model first
        if os.path.exists(IMPROVED_KMEANS_PATH):
            print(f"Using improved clustering model")
            model_blob = joblib.load(IMPROVED_KMEANS_PATH)
            X_ready, estimator = _prepare_features_for_model(model_blob, features)
            if hasattr(estimator, 'predict_cluster'):
                cluster = estimator.predict_cluster(X_ready)[0]
            else:
                cluster = estimator.predict(X_ready)[0]
            return cluster
        
        # Fallback to original model
        print(f"Using original KMeans clustering model")
        model_blob = joblib.load(KMEANS_MODEL_PATH)
        X_ready, estimator = _prepare_features_for_model(model_blob, features)
        if hasattr(estimator, 'predict_cluster'):
            cluster = estimator.predict_cluster(X_ready)[0]
        else:
            cluster = estimator.predict(X_ready)[0]
        return cluster
        
    except Exception as e:
        print(f"Error in cluster prediction: {e}")
        print(traceback.format_exc())
        # Return a default cluster if model fails
        return 0

if __name__ == '__main__':
    app.run(debug=True) 