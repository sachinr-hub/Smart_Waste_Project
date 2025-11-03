#!/usr/bin/env python3
"""
Script to fix all model issues, rename files, and ensure compatibility.
"""

import os
import shutil
import joblib
import traceback

def main():
    print("=" * 80)
    print("Fixing All Model Issues and File Names".center(80))
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "smart_waste_ml", "models", "saved_models")
    
    # Create backup directory
    backup_dir = os.path.join(project_root, "smart_waste_ml", "models", "backup_models")
    os.makedirs(backup_dir, exist_ok=True)
    
    # ===== Model File Names ======
    # Original models
    poly_reg_path = os.path.join(model_dir, 'polynomial_regression_model.joblib')
    log_reg_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
    old_knn_path = os.path.join(model_dir, 'kmeans_clustering_model.joblib')
    new_kmeans_path = os.path.join(model_dir, 'kmeans_clustering_model.joblib')
    
    # Improved models
    improved_poly_path = os.path.join(model_dir, 'improved_polynomial_regression_model.joblib')
    improved_log_path = os.path.join(model_dir, 'improved_logistic_regression_model.joblib')
    improved_kmeans_path = os.path.join(model_dir, 'improved_kmeans_clustering_model.joblib')
    
    try:
        # ===== Step 1: Rename KNN to KMeans if needed =====
        if os.path.exists(old_knn_path) and not os.path.exists(new_kmeans_path):
            print(f"Renaming {os.path.basename(old_knn_path)} to {os.path.basename(new_kmeans_path)}")
            
            # Backup first
            backup_path = os.path.join(backup_dir, os.path.basename(old_knn_path))
            shutil.copy2(old_knn_path, backup_path)
            
            # Copy to new name
            shutil.copy2(old_knn_path, new_kmeans_path)
            
            print(f"  ✓ File renamed successfully")
        
        # ===== Step 2: Create/update regular models =====
        # Polynomial Regression
        if os.path.exists(improved_poly_path):
            print(f"\nCopying improved polynomial regression model to standard filename")
            try:
                model_data = joblib.load(improved_poly_path)
                joblib.dump(model_data, poly_reg_path)
                print(f"  ✓ Polynomial regression model updated")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        # Logistic Regression
        if os.path.exists(improved_log_path):
            print(f"\nCopying improved logistic regression model to standard filename")
            try:
                model_data = joblib.load(improved_log_path)
                joblib.dump(model_data, log_reg_path)
                print(f"  ✓ Logistic regression model updated")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        # KMeans Clustering
        if os.path.exists(improved_kmeans_path):
            print(f"\nCopying improved kmeans clustering model to standard filename")
            try:
                model_data = joblib.load(improved_kmeans_path)
                joblib.dump(model_data, new_kmeans_path)
                print(f"  ✓ KMeans clustering model updated")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        print("\nAll model fixes complete! You can now run the application with `python run.py`")
        print("The old KNN model file can be safely deleted if desired.")
        
    except Exception as e:
        print(f"Error during model fixing: {str(e)}")
        print(traceback.format_exc())
        print("You can restore the original models from the backup directory if needed.")

if __name__ == "__main__":
    main() 