#!/usr/bin/env python3
"""
Script to rename remaining KNN files to KMeans for consistency.
"""

import os
import shutil

def main():
    print("=" * 80)
    print("Renaming Remaining KNN Files to KMeans".center(80))
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "smart_waste_ml", "models")
    saved_models_dir = os.path.join(models_dir, "saved_models")
    
    # Files to rename
    files_to_rename = [
        # Python module
        (os.path.join(models_dir, "kmeans_clustering.py"), 
         os.path.join(models_dir, "kmeans_clustering.py")),
         
        # Any plot files in metrics directory
        (os.path.join(models_dir, "metrics", "kmeans_clustering_plot.png"),
         os.path.join(models_dir, "metrics", "kmeans_clustering_plot.png"))
    ]
    
    # Files to delete
    files_to_delete = [
        os.path.join(saved_models_dir, "kmeans_clustering_model.joblib")
    ]
    
    # 1. Rename files
    for old_path, new_path in files_to_rename:
        if os.path.exists(old_path):
            print(f"Renaming {os.path.basename(old_path)} to {os.path.basename(new_path)}")
            shutil.copy2(old_path, new_path)
            os.remove(old_path)
            print(f"  ✓ File renamed successfully")
        else:
            print(f"  ℹ️ File not found: {old_path}")
    
    # 2. Delete files
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            print(f"Deleting {os.path.basename(file_path)}")
            os.remove(file_path)
            print(f"  ✓ File deleted successfully")
        else:
            print(f"  ℹ️ File not found: {file_path}")
    
    print("\nAll remaining KNN references have been updated to KMeans.")
    print("Your project now consistently uses KMeans clustering terminology.")

if __name__ == "__main__":
    main() 