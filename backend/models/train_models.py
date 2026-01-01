"""
Model Training Script
Trains machine learning models for student performance prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
from pathlib import Path

def load_data():
    """Load training data from CSV."""
    data_path = Path(__file__).parent.parent / 'data' / 'training_data.csv'
    df = pd.read_csv(data_path)
    return df

def prepare_features(df):
    """
    Prepare feature matrix from dataframe.
    
    Args:
        df: DataFrame with student data
        
    Returns:
        Feature matrix (X)
    """
    feature_columns = [
        'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
        'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm'
    ]
    
    X = df[feature_columns].values
    return X, feature_columns

def train_pass_fail_model(X_train, X_test, y_train, y_test):
    """Train Random Forest classifier for pass/fail prediction."""
    print("\n" + "="*60)
    print("Training Pass/Fail Classifier")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
    
    return model

def train_final_score_model(X_train, X_test, y_train, y_test):
    """Train Random Forest regressor for final exam score prediction."""
    print("\n" + "="*60)
    print("Training Final Exam Score Predictor")
    print("="*60)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRoot Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Average Prediction Error: ±{rmse:.2f} points")
    
    return model

def train_support_model(X_train, X_test, y_train, y_test):
    """Train Random Forest classifier for support need prediction."""
    print("\n" + "="*60)
    print("Training Support Need Classifier")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Support', 'Needs Support']))
    
    return model

def save_models(models, feature_names):
    """Save trained models to disk."""
    models_dir = Path(__file__).parent / 'saved_models'
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    joblib.dump(models['pass_fail'], models_dir / 'pass_fail_model.pkl')
    joblib.dump(models['final_score'], models_dir / 'final_score_model.pkl')
    joblib.dump(models['support'], models_dir / 'support_model.pkl')
    joblib.dump(feature_names, models_dir / 'feature_names.pkl')
    
    print("\n" + "="*60)
    print("Models Saved Successfully")
    print("="*60)
    print(f"Location: {models_dir}")
    print("Files:")
    print("  - pass_fail_model.pkl")
    print("  - final_score_model.pkl")
    print("  - support_model.pkl")
    print("  - feature_names.pkl")

def main():
    """Main training pipeline."""
    print("="*60)
    print("Student Performance Prediction - Model Training")
    print("="*60)
    
    # Load data
    print("\nLoading training data...")
    df = load_data()
    print(f"✓ Loaded {len(df)} records")
    
    # Prepare features
    X, feature_names = prepare_features(df)
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Prepare target variables
    y_pass_fail_train, y_pass_fail_test = train_test_split(
        df['pass_fail'].values, test_size=0.2, random_state=42
    )
    y_final_train, y_final_test = train_test_split(
        df['final_exam'].values, test_size=0.2, random_state=42
    )
    y_support_train, y_support_test = train_test_split(
        df['needs_support'].values, test_size=0.2, random_state=42
    )
    
    # Train models
    models = {}
    
    models['pass_fail'] = train_pass_fail_model(
        X_train, X_test, y_pass_fail_train, y_pass_fail_test
    )
    
    models['final_score'] = train_final_score_model(
        X_train, X_test, y_final_train, y_final_test
    )
    
    models['support'] = train_support_model(
        X_train, X_test, y_support_train, y_support_test
    )
    
    # Save models
    save_models(models, feature_names)
    
    print("\n✓ Training Complete!")

if __name__ == "__main__":
    main()
