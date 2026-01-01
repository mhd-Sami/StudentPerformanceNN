import joblib
import os
import sys

base_path = r"C:\Users\methe\.gemini\antigravity\scratch\student-performance-prediction\backend\models\saved_models"
nn_metrics_path = os.path.join(base_path, "nn_metrics.pkl")

try:
    if os.path.exists(nn_metrics_path):
        metrics = joblib.load(nn_metrics_path)
        print("Neural Network Metrics:", metrics)
    else:
        print("NN metrics file not found.")

except Exception as e:
    print(f"Error loading metrics: {e}")
