#!/usr/bin/env python3
"""
Script to train machine learning models with accuracy metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
    silhouette_score
)
import joblib
import random
import json

def main():
    print("=" * 80)
    print("Training and Evaluating Machine Learning Models".center(80))
    print("=" * 80)
    
    # Create directory structure if it doesn't exist
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "smart_waste_ml", "models", "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create metrics directory
    metrics_dir = os.path.join(project_root, "smart_waste_ml", "models", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate synthetic data for training
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    data = {
        'population_density': np.random.uniform(50, 1200, n_samples),
        'income_level': np.random.uniform(20000, 120000, n_samples),
        'recycling_rate': np.random.uniform(0.05, 0.85, n_samples),
        'public_awareness': np.random.uniform(1, 10, n_samples),
        'commercial_activity': np.random.uniform(1, 10, n_samples),
        'weather_temperature': np.random.uniform(0, 30, n_samples),
        'is_holiday': np.random.choice([0, 1], n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variables
    # 1. Waste generation (regression target)
    # Formula: base amount + factors for each feature
    waste_generation = (
        0.8 +  # Base amount
        0.0005 * df['population_density'] +
        0.000003 * df['income_level'] +
        -0.5 * df['recycling_rate'] +  # Higher recycling rate reduces waste
        -0.03 * df['public_awareness'] +  # Higher awareness reduces waste
        0.05 * df['commercial_activity'] +  # Commercial activity increases waste
        0.01 * df['weather_temperature'] +  # Higher temperature slightly increases waste
        0.1 * df['is_holiday'] +  # Holidays increase waste
        0.08 * df['is_weekend'] +  # Weekends increase waste
        np.sin(df['month'] * np.pi / 6) * 0.1 +  # Seasonal variation
        np.random.normal(0, 0.1, n_samples)  # Random noise
    )
    df['waste_generation'] = waste_generation
    
    # 2. Efficiency class (classification target, 0 to 3)
    efficiency_score = (
        10 -  # Start with maximum
        0.002 * df['population_density'] +  # Higher density slightly less efficient
        0.00001 * df['income_level'] +  # Higher income slightly more efficient
        5 * df['recycling_rate'] +  # Higher recycling much more efficient
        0.5 * df['public_awareness'] +  # Higher awareness more efficient
        -0.2 * df['commercial_activity'] +  # Commercial activity less efficient
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    # Convert to classes: 0 (Very Efficient), 1 (Efficient), 2 (Moderate), 3 (Inefficient)
    df['efficiency_class'] = pd.qcut(efficiency_score, 4, labels=False)
    
    print("Data generated successfully!")
    
    # Store model metrics
    metrics = {}
    
    # 1. Polynomial Regression Model
    print("\n" + "=" * 50)
    print("Polynomial Regression Model".center(50))
    print("=" * 50)
    reg_features = ['population_density', 'income_level', 'recycling_rate', 
                    'public_awareness', 'commercial_activity', 
                    'weather_temperature', 'is_holiday', 'is_weekend']
    
    X_reg = df[reg_features]
    y_reg = df['waste_generation']
    
    # Split into training and test sets
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Create and train polynomial regression model
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly_features.fit_transform(X_reg_train)
    X_poly_test = poly_features.transform(X_reg_test)
    
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_reg_train)
    
    # Evaluate model
    y_reg_pred = poly_reg.predict(X_poly_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, y_reg_pred)
    
    print("\nPolynomial Regression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R-squared (R²): {r2:.6f}")
    
    # Store metrics
    metrics['polynomial_regression'] = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }
    
    # Create a basic plot of predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
    plt.xlabel('Actual Waste Generation')
    plt.ylabel('Predicted Waste Generation')
    plt.title('Polynomial Regression: Actual vs Predicted')
    plt.savefig(os.path.join(metrics_dir, 'polynomial_regression_plot.png'))
    plt.close()
    
    # Save the model
    poly_reg_path = os.path.join(model_dir, 'polynomial_regression_model.joblib')
    poly_reg_data = {
        'model': poly_reg,
        'poly_features': poly_features,
        'feature_names': reg_features,
        'metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
    }
    joblib.dump(poly_reg_data, poly_reg_path)
    print(f"Polynomial Regression model saved to {poly_reg_path}")
    
    # 2. Logistic Regression Model
    print("\n" + "=" * 50)
    print("Logistic Regression Model".center(50))
    print("=" * 50)
    class_features = ['population_density', 'income_level', 'recycling_rate', 
                      'public_awareness', 'commercial_activity']
    
    X_class = df[class_features]
    y_class = df['efficiency_class']
    
    # Split into training and test sets
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    # Create and train logistic regression model
    log_reg = LogisticRegression(max_iter=1000, multi_class='multinomial')
    log_reg.fit(X_class_train, y_class_train)
    
    # Evaluate model
    y_class_pred = log_reg.predict(X_class_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_class_test, y_class_pred)
    precision = precision_score(y_class_test, y_class_pred, average='weighted')
    recall = recall_score(y_class_test, y_class_pred, average='weighted')
    f1 = f1_score(y_class_test, y_class_pred, average='weighted')
    
    print("\nLogistic Regression Metrics:")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    
    print("\nClassification Report:")
    report = classification_report(y_class_test, y_class_pred, target_names=['Very Efficient', 'Efficient', 'Moderate', 'Inefficient'])
    print(report)
    
    # Store metrics
    metrics['logistic_regression'] = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    # Create a confusion matrix
    cm = confusion_matrix(y_class_test, y_class_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['Very Efficient', 'Efficient', 'Moderate', 'Inefficient'], rotation=45)
    plt.yticks(tick_marks, ['Very Efficient', 'Efficient', 'Moderate', 'Inefficient'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(metrics_dir, 'logistic_regression_cm.png'))
    plt.close()
    
    # Save the model
    log_reg_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
    log_reg_data = {
        'model': log_reg,
        'feature_names': class_features,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    }
    joblib.dump(log_reg_data, log_reg_path)
    print(f"\nLogistic Regression model saved to {log_reg_path}")
    
    # 3. KNN Clustering Model
    print("\n" + "=" * 50)
    print("KNN Clustering Model".center(50))
    print("=" * 50)
    cluster_features = ['population_density', 'income_level', 'recycling_rate', 
                        'public_awareness', 'commercial_activity', 
                        'weather_temperature']
    
    X_cluster = df[cluster_features]
    
    # Standardize features for better clustering
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # Create and train KMeans clustering model
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    
    # Evaluate clustering
    silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    inertia = kmeans.inertia_
    
    print("\nKNN Clustering Metrics:")
    print(f"Silhouette Score: {silhouette_avg:.6f}")
    print(f"Inertia (Sum of squared distances): {inertia:.6f}")
    print(f"Number of clusters: 5")
    
    # Count samples in each cluster
    cluster_counts = np.bincount(cluster_labels)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i}: {count} samples ({count/len(cluster_labels)*100:.2f}%)")
    
    # Store metrics
    metrics['kmeans_clustering'] = {
        'silhouette_score': float(silhouette_avg),
        'inertia': float(inertia),
        'cluster_counts': [int(count) for count in cluster_counts],
        'cluster_percentages': [float(count/len(cluster_labels)*100) for count in cluster_counts]
    }
    
    # Create a scatter plot for the first two features
    plt.figure(figsize=(10, 8))
    
    # Reduce to two dimensions for visualization if needed
    # If you want better dimensionality reduction, you could use PCA or t-SNE here
    feature1, feature2 = 0, 1  # indices of the features to plot
    
    # Create scatter plot
    scatter = plt.scatter(
        X_cluster_scaled[:, feature1], 
        X_cluster_scaled[:, feature2], 
        c=cluster_labels, 
        cmap='viridis', 
        alpha=0.6
    )
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, feature1], 
        centroids[:, feature2], 
        marker='X', 
        s=200, 
        c='red', 
        label='Centroids'
    )
    
    plt.title('KNN Clustering Results')
    plt.xlabel(f'Scaled {cluster_features[feature1]}')
    plt.ylabel(f'Scaled {cluster_features[feature2]}')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'kmeans_clustering_plot.png'))
    plt.close()
    
    # Save the model
    knn_path = os.path.join(model_dir, 'kmeans_clustering_model.joblib')
    KMeans_data = {
        'model': kmeans,
        'scaler': scaler,
        'feature_names': cluster_features,
        'metrics': {
            'silhouette_score': float(silhouette_avg),
            'inertia': float(inertia),
            'n_clusters': 5
        }
    }
    joblib.dump(KMeans_data, knn_path)
    print(f"\nKNN Clustering model saved to {knn_path}")
    
    # Save all metrics to a JSON file
    metrics_path = os.path.join(metrics_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "=" * 80)
    print("Summary of Model Performance".center(80))
    print("=" * 80)
    
    print("\n1. Polynomial Regression:")
    print(f"   - R-squared (R²): {r2:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    
    print("\n2. Logistic Regression:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    
    print("\n3. KNN Clustering:")
    print(f"   - Silhouette Score: {silhouette_avg:.4f}")
    print(f"   - Number of clusters: 5")
    
    print("\nAll metrics have been saved to:")
    print(f"- {metrics_path}")
    print(f"- Visualizations in {metrics_dir}")
    
    print("\nAll models trained and evaluated successfully!")

if __name__ == "__main__":
    main() 