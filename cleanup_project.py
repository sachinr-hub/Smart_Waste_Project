#!/usr/bin/env python3
"""
Script to remove unwanted files and directories from the project.
"""

import os
import shutil
import sys

def main():
    print("Starting project cleanup...")
    
    # Define paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Files to be removed
    files_to_remove = [
        os.path.join(project_root, "test_models.py"),     # Testing script not needed in production
        os.path.join(project_root, "debug.py"),           # Debug script not needed in production
        os.path.join(project_root, "clean_project.py"),   # Old cleanup script
    ]
    
    # Directories to be removed
    dirs_to_remove = [
        os.path.join(project_root, "flask_waste_optimizer"),  # Old project directory
        os.path.join(project_root, "smart_waste_ml", "__pycache__"), # Compiled Python files
    ]
    
    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                print(f"Removing file: {file_path}")
                os.remove(file_path)
                print(f"Successfully removed {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                print(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path)
                print(f"Successfully removed {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")
    
    # Clean up __pycache__ directories
    for root, dirs, files in os.walk(project_root):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                try:
                    print(f"Removing Python cache directory: {pycache_path}")
                    shutil.rmtree(pycache_path)
                    print(f"Successfully removed {pycache_path}")
                except Exception as e:
                    print(f"Error removing directory {pycache_path}: {e}")
    
    # Optionally, keep recreate_models.py but rename to a more permanent name
    recreate_models_path = os.path.join(project_root, "recreate_models.py")
    model_training_path = os.path.join(project_root, "train_models.py")
    
    if os.path.exists(recreate_models_path):
        try:
            print(f"Renaming {recreate_models_path} to {model_training_path}")
            if os.path.exists(model_training_path):
                os.remove(model_training_path)
            os.rename(recreate_models_path, model_training_path)
            print("File renamed successfully")
        except Exception as e:
            print(f"Error renaming file: {e}")
    
    print("\nProject cleanup completed!")
    print("\nRemaining project files:")
    for root, dirs, files in os.walk(project_root):
        level = root.replace(project_root, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")
    
    print("\nTo run the application:")
    print("1. Ensure all requirements are installed:")
    print("   pip install -r smart_waste_ml/requirements.txt")
    print("2. Run the application:")
    print("   python run.py")
    print("3. Open your browser at http://127.0.0.1:5000/")

if __name__ == "__main__":
    main() 