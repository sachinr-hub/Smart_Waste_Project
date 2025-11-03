#!/usr/bin/env python3
"""
Script to fix model naming and format issues for compatibility.
"""

import os
import shutil
import joblib
import numpy as np
import traceback

def main():
    print("=" * 80)
    print("Fixing Model Format and Naming Issues".center(80))
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "smart_waste_ml", "models", "saved_models")
    
    # Create backup directory
    backup_dir = os.path.join(project_root, "smart_waste_ml", "models", "backup_models")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Check if improved models exist
    improved_poly_path = os.path.join(model_dir, 'improved_polynomial_regression_model.joblib')
    improved_log_path = os.path.join(model_dir, 'improved_logistic_regression_model.joblib')
    improved_kmeans_path = os.path.join(model_dir, 'improved_kmeans_clustering_model.joblib')
    
    # Target paths (used by app.py)
    poly_reg_path = os.path.join(model_dir, 'polynomial_regression_model.joblib')
    log_reg_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
    kmeans_path = os.path.join(model_dir, 'kmeans_clustering_model.joblib')
    
    try:
        # Backup existing models
        for path in [poly_reg_path, log_reg_path, kmeans_path]:
            if os.path.exists(path):
                backup_path = os.path.join(backup_dir, os.path.basename(path))
                print(f"Backing up {path} to {backup_path}")
                shutil.copy2(path, backup_path)
        
        # Process improved polynomial regression model
        if os.path.exists(improved_poly_path):
            print(f"Processing improved polynomial regression model...")
            model_data = joblib.load(improved_poly_path)
            # Ensure the model has the expected structure
            if 'model' in model_data:
                joblib.dump(model_data, poly_reg_path)
                print(f"Successfully copied improved polynomial model to {poly_reg_path}")
            else:
                print(f"Error: Improved polynomial model has unexpected format")
        else:
            print(f"Warning: Improved polynomial model not found at {improved_poly_path}")
            # Try to find it with any name
            for filename in os.listdir(model_dir):
                if 'polynomial' in filename.lower() and filename != os.path.basename(poly_reg_path):
                    improved_path = os.path.join(model_dir, filename)
                    print(f"Found potential polynomial model: {improved_path}")
                    try:
                        model_data = joblib.load(improved_path)
                        joblib.dump(model_data, poly_reg_path)
                        print(f"Successfully copied {filename} to {poly_reg_path}")
                        break
                    except Exception as e:
                        print(f"Error loading model {filename}: {str(e)}")
        
        # Process improved logistic regression model
        if os.path.exists(improved_log_path):
            print(f"Processing improved logistic regression model...")
            model_data = joblib.load(improved_log_path)
            # Ensure the model has the expected structure
            if 'model' in model_data:
                joblib.dump(model_data, log_reg_path)
                print(f"Successfully copied improved logistic model to {log_reg_path}")
            else:
                print(f"Error: Improved logistic model has unexpected format")
        else:
            print(f"Warning: Improved logistic model not found at {improved_log_path}")
            # Try to find it with any name
            for filename in os.listdir(model_dir):
                if 'logistic' in filename.lower() and filename != os.path.basename(log_reg_path):
                    improved_path = os.path.join(model_dir, filename)
                    print(f"Found potential logistic model: {improved_path}")
                    try:
                        model_data = joblib.load(improved_path)
                        joblib.dump(model_data, log_reg_path)
                        print(f"Successfully copied {filename} to {log_reg_path}")
                        break
                    except Exception as e:
                        print(f"Error loading model {filename}: {str(e)}")
        
        # Process improved clustering model
        if os.path.exists(improved_kmeans_path):
            print(f"Processing improved clustering model...")
            model_data = joblib.load(improved_kmeans_path)
            # Ensure the model has the expected structure
            if 'model' in model_data:
                joblib.dump(model_data, kmeans_path)
                print(f"Successfully copied improved clustering model to {kmeans_path}")
            else:
                print(f"Error: Improved clustering model has unexpected format")
        else:
            print(f"Warning: Improved clustering model not found at {improved_kmeans_path}")
            # Try to find it with any name
            for filename in os.listdir(model_dir):
                if ('cluster' in filename.lower() or 'kmeans' in filename.lower()) and filename != os.path.basename(kmeans_path):
                    improved_path = os.path.join(model_dir, filename)
                    print(f"Found potential clustering model: {improved_path}")
                    try:
                        model_data = joblib.load(improved_path)
                        joblib.dump(model_data, kmeans_path)
                        print(f"Successfully copied {filename} to {kmeans_path}")
                        break
                    except Exception as e:
                        print(f"Error loading model {filename}: {str(e)}")
        
        print("\nModel fixes complete! You can now run the application with `python run.py`")
        print("If issues persist, the original models have been backed up to:", backup_dir)
        
    except Exception as e:
        print(f"Error during model fixing: {str(e)}")
        print(traceback.format_exc())
        print("You can restore the original models from the backup directory if needed.")

if __name__ == "__main__":
    main() 