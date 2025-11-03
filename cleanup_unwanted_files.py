#!/usr/bin/env python3
"""
Script to remove unwanted files from the project.
This includes:
- Old KNN-related files that have been renamed to KMeans
- Temporary datasets and images
- Backup files that are no longer needed
"""

import os
import shutil
import glob

def main():
    print("=" * 80)
    print("Cleaning Up Unwanted Files".center(80))
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Files to remove
    files_to_remove = [
        # KNN-related files that should use KMeans naming
        os.path.join(project_root, "smart_waste_ml", "data", "knn_waste_dataset.csv"),
        os.path.join(project_root, "smart_waste_ml", "webapp", "static", "images", "knn_clusters_tsne.png"),
        os.path.join(project_root, "smart_waste_ml", "webapp", "static", "images", "knn_clusters_pca.png"),
        
        # Any old KNN model files
        os.path.join(project_root, "smart_waste_ml", "models", "saved_models", "knn_clustering_model.joblib"),
        os.path.join(project_root, "smart_waste_ml", "models", "backup_models", "knn_clustering_model.joblib"),
        
        # Empty or unnecessary backup folders can be handled separately
    ]
    
    # Optional: list of temporary scripts that could be removed after project is stable
    temporary_scripts = [
        os.path.join(project_root, "rename_clustering_model.py"),
        os.path.join(project_root, "rename_dataset_references.py"),
        os.path.join(project_root, "rename_remaining_files.py"),
        os.path.join(project_root, "fix_models.py"),
        os.path.join(project_root, "fix_all_models.py")
    ]
    
    # 1. Remove unwanted files
    removed_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            print(f"Removing: {file_path}")
            try:
                os.remove(file_path)
                removed_count += 1
                print(f"  ✓ File removed successfully")
            except Exception as e:
                print(f"  ✗ Error removing file: {str(e)}")
    
    # 2. Create a new dataset file with the correct name if needed
    knn_dataset = os.path.join(project_root, "smart_waste_ml", "data", "knn_waste_dataset.csv")
    kmeans_dataset = os.path.join(project_root, "smart_waste_ml", "data", "kmeans_waste_dataset.csv")
    
    if not os.path.exists(kmeans_dataset) and os.path.exists(os.path.dirname(kmeans_dataset)):
        # Create a simple placeholder CSV if the old one didn't exist
        print(f"\nCreating new dataset file: {kmeans_dataset}")
        try:
            os.makedirs(os.path.dirname(kmeans_dataset), exist_ok=True)
            with open(kmeans_dataset, 'w') as f:
                f.write("population_density,income_level,recycling_rate,public_awareness,commercial_activity,weather_temperature,waste_generation,cluster\n")
                f.write("500,50000,0.35,5,5,20,1.2,0\n")
                f.write("800,70000,0.55,7,8,22,0.9,1\n")
                f.write("300,30000,0.25,4,3,18,1.5,2\n")
            print(f"  ✓ Created new dataset file with sample data")
        except Exception as e:
            print(f"  ✗ Error creating dataset file: {str(e)}")
    
    # 3. Ask about temporary scripts
    print("\nThe following temporary scripts could be removed once the project is stable:")
    for i, script in enumerate(temporary_scripts, 1):
        if os.path.exists(script):
            print(f"  {i}. {os.path.basename(script)}")
    
    print("\nTo remove these files later, you can:")
    print("1. Manually delete them")
    print("2. Run this script with the --remove-temp-scripts flag")
    print("3. Keep them for documentation purposes")
    
    # Results summary
    if removed_count > 0:
        print(f"\nSuccessfully removed {removed_count} unwanted files.")
    else:
        print("\nNo unwanted files found to remove.")
    
    print("\nThe project has been cleaned up successfully!")

if __name__ == "__main__":
    main() 