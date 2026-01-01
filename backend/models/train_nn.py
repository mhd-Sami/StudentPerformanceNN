"""
Neural Network Training Script (scikit-learn MLP)
Trains neural network models using scikit-learn's MLPClassifier and MLPRegressor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib

# Set random seed
np.random.seed(42)

def load_data():
    """Load training data from CSV."""
    data_path = Path(__file__).parent.parent / 'data' / 'training_data.csv'
    df = pd.read_csv(data_path)
    return df

def engineer_features(df):
    """
    Create engineered features from raw data.
    
    Args:
        df: DataFrame with student data
        
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # Average scores
    df['quiz_avg'] = df[['quiz1', 'quiz2', 'quiz3', 'quiz4']].mean(axis=1)
    df['assignment_avg'] = df[['assignment1', 'assignment2', 'assignment3', 'assignment4']].mean(axis=1)
    
    # Overall average (excluding final exam)
    df['overall_avg'] = df[['quiz_avg', 'assignment_avg', 'midterm']].mean(axis=1)
    
    # Performance trends
    df['quiz_trend'] = df['quiz4'] - df['quiz1']
    df['assignment_trend'] = df['assignment4'] - df['assignment1']
    
    # Consistency (standard deviation)
    df['quiz_std'] = df[['quiz1', 'quiz2', 'quiz3', 'quiz4']].std(axis=1)
    df['assignment_std'] = df[['assignment1', 'assignment2', 'assignment3', 'assignment4']].std(axis=1)
    
    # Attendance impact
    df['attendance_score_interaction'] = df['attendance'] * df['overall_avg'] / 100
    
    # Performance categories
    df['high_performer'] = (df['overall_avg'] >= 80).astype(int)
    df['low_performer'] = (df['overall_avg'] < 60).astype(int)
    
    return df

def prepare_features(df):
    """
    Prepare feature matrix from dataframe.
    
    Args:
        df: DataFrame with student data
        
    Returns:
        Feature matrix (X) and feature names
    """
    # Original features
    base_features = [
        'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
        'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm'
    ]
    
    # Engineered features
    engineered_features = [
        'quiz_avg', 'assignment_avg', 'overall_avg',
        'quiz_trend', 'assignment_trend',
        'quiz_std', 'assignment_std',
        'attendance_score_interaction',
        'high_performer', 'low_performer'
    ]
    
    feature_columns = base_features + engineered_features
    X = df[feature_columns].values
    
    return X, feature_columns

def train_pass_fail_model(X_train, X_test, y_train, y_test):
    """Train MLP classifier for pass/fail prediction."""
    print("\n" + "="*60)
    print("Training Pass/Fail Neural Network Classifier")
    print("="*60)
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
    
    return model

def train_final_score_model(X_train, X_test, y_train, y_test):
    """Train MLP regressor for final exam score prediction."""
    print("\n" + "="*60)
    print("Training Final Exam Score Neural Network Predictor")
    print("="*60)
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
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
    """Train MLP classifier for support need prediction."""
    print("\n" + "="*60)
    print("Training Support Need Neural Network Classifier")
    print("="*60)
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Support', 'Needs Support']))
    
    return model

def save_models(models, scaler, feature_names):
    """Save trained models and preprocessing objects."""
    models_dir = Path(__file__).parent / 'saved_models'
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    joblib.dump(models['pass_fail'], models_dir / 'pass_fail_nn.pkl')
    joblib.dump(models['final_score'], models_dir / 'final_score_nn.pkl')
    joblib.dump(models['support'], models_dir / 'support_nn.pkl')
    
    # Save scaler and feature names
    joblib.dump(scaler, models_dir / 'scaler_nn.pkl')
    joblib.dump(feature_names, models_dir / 'feature_names_nn.pkl')
    
    print("\n" + "="*60)
    print("Neural Network Models Saved Successfully")
    print("="*60)
    print(f"Location: {models_dir}")
    print("Files:")
    print("  - pass_fail_nn.pkl")
    print("  - final_score_nn.pkl")
    print("  - support_nn.pkl")
    print("  - scaler_nn.pkl")
    print("  - feature_names_nn.pkl")

def main():
    """Main training pipeline."""
    print("="*60)
    print("Neural Network Training - Student Performance Prediction")
    print("="*60)
    
    # Load data
    print("\nLoading training data...")
    df = load_data()
    print(f"✓ Loaded {len(df)} records")
    
    # Engineer features
    print("\nEngineering features...")
    df_engineered = engineer_features(df)
    print(f"✓ Created {len(df_engineered.columns) - len(df.columns)} new features")
    
    # Prepare features
    X, feature_names = prepare_features(df_engineered)
    print(f"✓ Total features: {len(feature_names)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for final training
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # Prepare targets
    y_pass_fail = df_engineered['pass_fail'].values
    y_final = df_engineered['final_exam'].values
    y_support = df_engineered['needs_support'].values
    
    y_pass_fail_train, y_pass_fail_test = train_test_split(y_pass_fail, test_size=0.2, random_state=42)
    y_final_train, y_final_test = train_test_split(y_final, test_size=0.2, random_state=42)
    y_support_train, y_support_test = train_test_split(y_support, test_size=0.2, random_state=42)
    
    # Train models
    models = {}
    model_metrics = {}
    
    # --- 1. Pass/Fail Classifier ---
    print("\n" + "="*60)
    print("1. Pass/Fail Classifier")
    print("="*60)
    
    # Cross-Validation
    from sklearn.model_selection import cross_val_score
    pf_model_cv = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
    cv_scores = cross_val_score(pf_model_cv, X_scaled, y_pass_fail, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    models['pass_fail'] = train_pass_fail_model(
        X_train, X_test, y_pass_fail_train, y_pass_fail_test
    )
    
    # Feature Importance (Permutation)
    from sklearn.inspection import permutation_importance
    print("Calculating feature importance...")
    pf_perm_importance = permutation_importance(models['pass_fail'], X_test, y_pass_fail_test, n_repeats=10, random_state=42)
    
    # --- 2. Final Score Predictor ---
    print("\n" + "="*60)
    print("2. Final Exam Score Predictor")
    print("="*60)
    
    # Cross-Validation
    fs_model_cv = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
    cv_scores_rmse = cross_val_score(fs_model_cv, X_scaled, y_final, cv=5, scoring='neg_root_mean_squared_error')
    print(f"Cross-Validation RMSE: {-cv_scores_rmse.mean():.2f} (+/- {cv_scores_rmse.std():.2f})")
    
    # Store RMSE for confidence intervals
    model_metrics['final_score_rmse'] = -cv_scores_rmse.mean()
    
    models['final_score'] = train_final_score_model(
        X_train, X_test, y_final_train, y_final_test
    )
    
    # Feature Importance (Permutation)
    print("Calculating feature importance...")
    fs_perm_importance = permutation_importance(models['final_score'], X_test, y_final_test, n_repeats=10, random_state=42)

    # --- 3. Support Need Classifier ---
    print("\n" + "="*60)
    print("3. Support Need Classifier")
    print("="*60)
    
    # Cross-Validation
    sn_model_cv = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
    cv_scores = cross_val_score(sn_model_cv, X_scaled, y_support, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    models['support'] = train_support_model(
        X_train, X_test, y_support_train, y_support_test
    )

    # Save models and metadata
    save_models(models, scaler, feature_names)
    
    # Save extra metadata for UI
    models_dir = Path(__file__).parent / 'saved_models'
    
    # Save metrics
    joblib.dump(model_metrics, models_dir / 'nn_metrics.pkl')
    
    # Save Feature Importance (averaged across models or just use Final Score importance as representative)
    # We'll save the Final Score importance as it's most granular
    feature_importance_dict = dict(zip(feature_names, fs_perm_importance.importances_mean))
    joblib.dump(feature_importance_dict, models_dir / 'nn_feature_importance.pkl')
    
    print("\n✓ Training Complete! Saved models, metrics, and feature importance.")

if __name__ == "__main__":
    main()
