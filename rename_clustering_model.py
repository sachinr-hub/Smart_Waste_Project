#!/usr/bin/env python3
"""
Script to rename KNN clustering model to KMeans clustering model for accuracy.
"""

import os
import shutil
import fileinput
import sys

def main():
    print("=" * 80)
    print("Renaming KNN Clustering Model to KMeans Clustering Model".center(80))
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "smart_waste_ml", "models", "saved_models")
    
    # Original and new filenames
    old_name = "kmeans_clustering_model.joblib"
    new_name = "kmeans_clustering_model.joblib"
    
    old_path = os.path.join(model_dir, old_name)
    new_path = os.path.join(model_dir, new_name)
    
    # Create backup 
    backup_dir = os.path.join(project_root, "smart_waste_ml", "models", "backup_models")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files to update references in
    files_to_update = [
        os.path.join(project_root, "smart_waste_ml", "webapp", "app.py"),
        os.path.join(project_root, "fix_models.py"),
        os.path.join(project_root, "test_project.py")
    ]
    
    # Rename model file
    if os.path.exists(old_path):
        # Backup the model
        backup_path = os.path.join(backup_dir, old_name)
        print(f"Creating backup of {old_name} in {backup_dir}")
        shutil.copy2(old_path, backup_path)
        
        # Rename the file
        print(f"Renaming {old_name} to {new_name}")
        shutil.copy2(old_path, new_path)
        
        # Update references in files
        for file_path in files_to_update:
            if os.path.exists(file_path):
                print(f"Updating references in {file_path}")
                
                with fileinput.FileInput(file_path, inplace=True) as file:
                    for line in file:
                        # Replace filename occurrences but preserve indentation
                        line = line.replace(old_name, new_name)
                        # Also replace variable names to match
                        line = line.replace("KNN_MODEL_PATH", "KMEANS_MODEL_PATH")
                        line = line.replace("knn_path", "kmeans_path")
                        line = line.replace("knn_model", "kmeans_model")
                        line = line.replace("KMeans_data", "KMeans_data")
                        sys.stdout.write(line)
        
        print("\nAll files updated successfully!")
        print(f"You can now delete the old file: {old_path}")
        print("Don't forget to commit these changes to your version control system.")
    else:
        print(f"Error: Original model file {old_path} not found.")
        print("No changes were made.")
    
if __name__ == "__main__":
    main() 