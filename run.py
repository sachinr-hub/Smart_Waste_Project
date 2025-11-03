#!/usr/bin/env python3
"""
Waste Management ML Predictor
Main entry point for the application
"""

import os
import sys
import traceback

try:
    print("=" * 80)
    print("Waste Management ML Predictor".center(80))
    print("=" * 80)
    print("\nStarting web application...\n")
    
    # Import app here to catch import errors
    from smart_waste_ml.webapp.app import app
    
    print("Open your browser at http://127.0.0.1:5000/")
    print("Use the application to make predictions with:")
    print("- Polynomial Regression (Waste Generation)")
    print("- Logistic Regression (Efficiency Classification)")
    print("- KMeans Clustering (Neighborhood Grouping)")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True)
except Exception as e:
    print("Error starting application:")
    print(str(e))
    print("\nTraceback:")
    traceback.print_exc()
    input("Press Enter to exit...") 