"""
Feature Engineering Module
Creates advanced features from raw student data for better ML predictions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class FeatureEngineer:
    """Advanced feature engineering for student performance prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
    def create_statistical_features(self, df):
        """Create statistical aggregation features."""
        features = df.copy()
        
        # Quiz scores
        quiz_cols = ['quiz1', 'quiz2', 'quiz3', 'quiz4']
        features['quiz_mean'] = df[quiz_cols].mean(axis=1)
        features['quiz_std'] = df[quiz_cols].std(axis=1)
        features['quiz_min'] = df[quiz_cols].min(axis=1)
        features['quiz_max'] = df[quiz_cols].max(axis=1)
        features['quiz_range'] = features['quiz_max'] - features['quiz_min']
        
        # Assignment scores
        assign_cols = ['assignment1', 'assignment2', 'assignment3', 'assignment4']
        features['assignment_mean'] = df[assign_cols].mean(axis=1)
        features['assignment_std'] = df[assign_cols].std(axis=1)
        features['assignment_min'] = df[assign_cols].min(axis=1)
        features['assignment_max'] = df[assign_cols].max(axis=1)
        features['assignment_range'] = features['assignment_max'] - features['assignment_min']
        
        # Overall statistics
        all_scores = quiz_cols + assign_cols + ['midterm']
        features['overall_mean'] = df[all_scores].mean(axis=1)
        features['overall_std'] = df[all_scores].std(axis=1)
        features['overall_median'] = df[all_scores].median(axis=1)
        
        return features
    
    def create_trend_features(self, df):
        """Create trend and progression features."""
        features = df.copy()
        
        # Quiz trend (linear regression slope)
        quiz_cols = ['quiz1', 'quiz2', 'quiz3', 'quiz4']
        quiz_values = df[quiz_cols].values
        x = np.arange(4)
        
        quiz_trends = []
        for row in quiz_values:
            slope = np.polyfit(x, row, 1)[0]
            quiz_trends.append(slope)
        
        features['quiz_trend'] = quiz_trends
        
        # Assignment trend
        assign_cols = ['assignment1', 'assignment2', 'assignment3', 'assignment4']
        assign_values = df[assign_cols].values
        
        assign_trends = []
        for row in assign_values:
            slope = np.polyfit(x, row, 1)[0]
            assign_trends.append(slope)
        
        features['assignment_trend'] = assign_trends
        
        # Improvement rate (midterm vs early performance)
        early_avg = (features['quiz_mean'] + features['assignment_mean']) / 2
        features['improvement_rate'] = (df['midterm'] - early_avg) / (early_avg + 1e-6)
        
        # Consistency score (inverse of std dev)
        features['consistency_score'] = 1 / (features['overall_std'] + 1e-6)
        
        return features
    
    def create_interaction_features(self, df):
        """Create interaction and derived features."""
        features = df.copy()
        
        # Attendance interactions
        features['attendance_performance'] = df['attendance'] * features['overall_mean']
        features['attendance_quiz'] = df['attendance'] * features['quiz_mean']
        features['attendance_assignment'] = df['attendance'] * features['assignment_mean']
        
        # Performance gaps
        features['quiz_assignment_gap'] = abs(features['quiz_mean'] - features['assignment_mean'])
        features['midterm_avg_gap'] = abs(df['midterm'] - features['overall_mean'])
        
        # Weighted averages (simulating actual grading)
        features['weighted_score'] = (
            df['attendance'] * 0.1 +
            features['quiz_mean'] * 0.2 +
            features['assignment_mean'] * 0.3 +
            df['midterm'] * 0.4
        )
        
        # Performance categories (binning)
        features['performance_category'] = pd.cut(
            features['overall_mean'],
            bins=[0, 50, 70, 85, 100],
            labels=[0, 1, 2, 3]  # Poor, Average, Good, Excellent
        ).astype(int)
        
        features['attendance_category'] = pd.cut(
            df['attendance'],
            bins=[0, 60, 80, 100],
            labels=[0, 1, 2]  # Low, Medium, High
        ).astype(int)
        
        return features
    
    def create_polynomial_features(self, df, base_features):
        """Create polynomial interaction features."""
        # Select key features for polynomial expansion
        key_features = [
            'attendance', 'quiz_mean', 'assignment_mean', 'midterm',
            'overall_mean', 'quiz_trend', 'assignment_trend'
        ]
        
        X_poly = self.poly.fit_transform(df[key_features])
        poly_feature_names = self.poly.get_feature_names_out(key_features)
        
        # Add polynomial features to dataframe
        poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)
        
        # Remove original features (already in df) and keep only interactions
        interaction_cols = [col for col in poly_df.columns if ' ' in col]
        
        return pd.concat([df, poly_df[interaction_cols]], axis=1)
    
    def engineer_features(self, df, include_poly=True):
        """
        Main feature engineering pipeline.
        
        Args:
            df: DataFrame with raw features
            include_poly: Whether to include polynomial features
            
        Returns:
            DataFrame with engineered features
        """
        # Start with original features
        features = df.copy()
        
        # Add statistical features
        features = self.create_statistical_features(features)
        
        # Add trend features
        features = self.create_trend_features(features)
        
        # Add interaction features
        features = self.create_interaction_features(features)
        
        # Optionally add polynomial features
        if include_poly:
            base_features = list(features.columns)
            features = self.create_polynomial_features(features, base_features)
        
        return features
    
    def get_feature_names(self, df):
        """Get list of all engineered feature names."""
        engineered = self.engineer_features(df, include_poly=False)
        return list(engineered.columns)
    
    def scale_features(self, X_train, X_test=None):
        """
        Standardize features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled


def create_engineered_features(df):
    """
    Convenience function to create engineered features.
    
    Args:
        df: DataFrame with raw student data
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(df, include_poly=True)


if __name__ == "__main__":
    # Test feature engineering
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent / 'data' / 'training_data.csv'
    df = pd.read_csv(data_path)
    
    print("Original features:", df.shape[1])
    
    engineer = FeatureEngineer()
    engineered = engineer.engineer_features(df, include_poly=True)
    
    print("Engineered features:", engineered.shape[1])
    print("\nNew features created:")
    
    original_cols = set(df.columns)
    new_cols = set(engineered.columns) - original_cols
    
    for col in sorted(new_cols):
        print(f"  - {col}")
