"""
Neural Network Predictor (scikit-learn MLP)
Loads and uses trained MLP neural network models for predictions.
"""

import numpy as np
import joblib
import pandas as pd
from pathlib import Path

class NeuralNetworkPredictor:
    """Predictor using trained MLP neural network models."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.models_loaded = False
        
    def load_models(self):
        """Load trained neural network models from disk."""
        models_dir = Path(__file__).parent / 'saved_models'
        
        try:
            # Load models
            self.models['pass_fail'] = joblib.load(models_dir / 'pass_fail_nn.pkl')
            self.models['final_score'] = joblib.load(models_dir / 'final_score_nn.pkl')
            self.models['support'] = joblib.load(models_dir / 'support_nn.pkl')
            
            # Load scaler and feature names
            self.scaler = joblib.load(models_dir / 'scaler_nn.pkl')
            self.feature_names = joblib.load(models_dir / 'feature_names_nn.pkl')
            
            self.models_loaded = True
            print("✓ Neural network models loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load neural network models: {e}")
            self.models_loaded = False
            raise
    
    def engineer_features(self, df):
        """Create engineered features from raw data."""
        df = df.copy()
        
        # Average scores
        df['quiz_avg'] = df[['quiz1', 'quiz2', 'quiz3', 'quiz4']].mean(axis=1)
        df['assignment_avg'] = df[['assignment1', 'assignment2', 'assignment3', 'assignment4']].mean(axis=1)
        
        # Overall average
        df['overall_avg'] = df[['quiz_avg', 'assignment_avg', 'midterm']].mean(axis=1)
        
        # Performance trends
        df['quiz_trend'] = df['quiz4'] - df['quiz1']
        df['assignment_trend'] = df['assignment4'] - df['assignment1']
        
        # Consistency
        df['quiz_std'] = df[['quiz1', 'quiz2', 'quiz3', 'quiz4']].std(axis=1)
        df['assignment_std'] = df[['assignment1', 'assignment2', 'assignment3', 'assignment4']].std(axis=1)
        
        # Attendance impact
        df['attendance_score_interaction'] = df['attendance'] * df['overall_avg'] / 100
        
        # Performance categories
        df['high_performer'] = (df['overall_avg'] >= 80).astype(int)
        df['low_performer'] = (df['overall_avg'] < 60).astype(int)
        
        return df
    
    def prepare_features(self, student_data):
        """
        Prepare features from raw student data.
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Scaled feature array ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([student_data])
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Select features
        X = df_engineered[self.feature_names].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, student_data):
        """
        Make predictions using neural network models.
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with predictions
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Prepare features
        X = self.prepare_features(student_data)
        
        # Get feature values for display
        df = pd.DataFrame([student_data])
        df_engineered = self.engineer_features(df)
        feature_values = df_engineered[self.feature_names].iloc[0].to_dict()
        
        # Make predictions
        pass_fail_pred = self.models['pass_fail'].predict(X)[0]
        pass_fail_proba = self.models['pass_fail'].predict_proba(X)[0]
        
        final_score_pred = self.models['final_score'].predict(X)[0]
        
        support_pred = self.models['support'].predict(X)[0]
        support_proba = self.models['support'].predict_proba(X)[0]
        
        # Calculate confidence scores
        pass_confidence = max(pass_fail_proba) * 100
        support_confidence = max(support_proba) * 100
        
        # Prepare results
        results = {
            'pass_fail': {
                'prediction': 'Pass' if pass_fail_pred == 1 else 'Fail',
                'confidence': round(pass_confidence, 2),
                'probability_pass': round(pass_fail_proba[1] * 100, 2),
                'probability_fail': round(pass_fail_proba[0] * 100, 2)
            },
            'final_exam_score': {
                'predicted_score': round(final_score_pred, 2),
                'grade': self._get_grade(final_score_pred)
            },
            'support_needed': {
                'prediction': 'Yes' if support_pred == 1 else 'No',
                'confidence': round(support_confidence, 2),
                'recommendation': self._get_recommendation(support_pred, final_score_pred)
            },
            'overall_assessment': self._get_overall_assessment(
                pass_fail_pred, final_score_pred, support_pred
            ),
            'features_used': feature_values,
            'num_features': len(self.feature_names)
        }
        
        return results
    
    def _get_grade(self, score):
        """Convert numerical score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        elif score >= 50:
            return 'E'
        else:
            return 'F'
    
    def _get_recommendation(self, support_pred, final_score):
        """Generate recommendation based on predictions."""
        if support_pred == 1:
            if final_score < 50:
                return "Immediate intervention required. Consider tutoring and extra study sessions."
            elif final_score < 60:
                return "Additional support recommended. Focus on weak areas before final exam."
            else:
                return "Some support may help. Review difficult topics and practice more."
        else:
            if final_score >= 80:
                return "Excellent performance! Keep up the good work."
            elif final_score >= 70:
                return "Good performance. Continue current study habits."
            else:
                return "Satisfactory performance. Minor improvements possible."
    
    def _get_overall_assessment(self, pass_fail, final_score, support):
        """Generate overall assessment."""
        if pass_fail == 1 and final_score >= 80:
            return "Strong Performance - Student is on track for excellent results."
        elif pass_fail == 1 and final_score >= 60:
            return "Satisfactory Performance - Student should pass with adequate preparation."
        elif pass_fail == 1:
            return "Borderline Performance - Student may pass but needs focused effort."
        else:
            return "At Risk - Student needs immediate support to improve chances of passing."

# Global instance
_nn_predictor = None

def get_nn_predictor():
    """Get or create neural network predictor instance."""
    global _nn_predictor
    if _nn_predictor is None:
        _nn_predictor = NeuralNetworkPredictor()
        try:
            _nn_predictor.load_models()
        except Exception as e:
            print(f"Could not load NN models: {e}")
            _nn_predictor = None
            raise
    return _nn_predictor
