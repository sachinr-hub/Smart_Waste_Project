#!/usr/bin/env python3
"""
Script to find and rename any dataset references from KNN to KMeans.
This script searches through Python files and JSON metrics files.
"""

import os
import json
import re
import glob

def main():
    print("=" * 80)
    print("Renaming KNN Dataset References to KMeans".center(80))
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Directories to search
    search_paths = [
        os.path.join(project_root, "*.py"),
        os.path.join(project_root, "smart_waste_ml", "models", "*.py"),
        os.path.join(project_root, "smart_waste_ml", "models", "metrics", "*.json"),
        os.path.join(project_root, "smart_waste_ml", "webapp", "*.py"),
        os.path.join(project_root, "smart_waste_ml", "data", "*.csv"),
        os.path.join(project_root, "smart_waste_ml", "data", "*.json")
    ]
    
    # Find all relevant files
    all_files = []
    for path in search_paths:
        all_files.extend(glob.glob(path))
    
    # Patterns to look for in the code
    patterns = [
        (r'KMeans_dataset', 'KMeans_dataset'),
        (r'KMeans_data', 'KMeans_data'),
        (r'KMeans_data', 'KMeans_data'),
        (r'"kmeans":', '"kmeans":'),
        (r"'kmeans':", "'kmeans':"),
        (r'kmeans_features', 'kmeans_features'),
        (r'kmeans_df', 'kmeans_df'),
        (r'kmeans_cluster', 'kmeans_cluster'),
        (r'cluster_kmeans', 'cluster_kmeans')
    ]
    
    # Special handling for JSON files
    def update_json_file(filepath):
        print(f"Checking JSON file: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check if there's a 'knn' key
            if 'knn' in data:
                data['kmeans'] = data.pop('knn')
                print(f"  ✓ Renamed 'knn' key to 'kmeans' in {os.path.basename(filepath)}")
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
                return True
            
            # Check in clustering key
            if 'clustering' in data:
                if 'model_type' in data['clustering'] and data['clustering']['model_type'] == 'kmeans':
                    data['clustering']['model_type'] = 'kmeans'
                    print(f"  ✓ Updated clustering model_type from 'knn' to 'kmeans'")
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=4)
                    return True
            
            return False
        except Exception as e:
            print(f"  ✗ Error processing JSON file {filepath}: {str(e)}")
            return False
    
    # Process files
    files_updated = 0
    updated_files = []
    
    for filepath in all_files:
        if filepath.endswith('.json'):
            if update_json_file(filepath):
                files_updated += 1
                updated_files.append(filepath)
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            updated_content = content
            found_match = False
            pattern_matches = []
            
            for pattern, replacement in patterns:
                if re.search(pattern, updated_content, re.IGNORECASE):
                    found_match = True
                    match_count = len(re.findall(pattern, updated_content, re.IGNORECASE))
                    pattern_matches.append(f"'{pattern}' ({match_count} occurrences)")
                    updated_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            
            if found_match:
                print(f"Updating file: {filepath}")
                print(f"  Found patterns: {', '.join(pattern_matches)}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                files_updated += 1
                updated_files.append(filepath)
                print(f"  ✓ File updated successfully")
        except Exception as e:
            print(f"  ✗ Error processing file {filepath}: {str(e)}")
    
    if files_updated > 0:
        print(f"\nSuccessfully updated {files_updated} files:")
        for file in updated_files:
            print(f"  - {file}")
    else:
        print("\nNo files needed updating. All dataset references appear to be using KMeans already.")
    
    print("\nAll KNN dataset references have been updated to KMeans.")

if __name__ == "__main__":
    main() 