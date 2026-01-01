"""
Prediction Module
Loads trained models and provides prediction interface.
"""

import joblib
import numpy as np
from pathlib import Path

class StudentPerformancePredictor:
    """Predictor class for student performance."""
    
    def __init__(self):
        """Initialize predictor by loading trained models."""
        self.models_dir = Path(__file__).parent / 'saved_models'
        self.load_models()
        self.scaler = None
        # Try loading scaler if exists
        try:
             self.scaler = joblib.load(self.models_dir / 'scaler_rf.pkl')
        except:
             pass
    
    def load_models(self):
        """Load all trained models from disk."""
        try:
            self.pass_fail_model = joblib.load(self.models_dir / 'pass_fail_model.pkl')
            self.final_score_model = joblib.load(self.models_dir / 'final_score_model.pkl')
            self.support_model = joblib.load(self.models_dir / 'support_model.pkl')
            self.feature_names = joblib.load(self.models_dir / 'feature_names.pkl')
            print("✓ Models loaded successfully")
        except FileNotFoundError as e:
            raise Exception(
                "Models not found. Please run 'python backend/models/train_models.py' first."
            ) from e

    def engineer_features(self, df):
        """Create engineered features from raw data (Same as NN)."""
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

    def prepare_input(self, student_data):
        """
        Prepare input data for prediction with engineering.
        """
        import pandas as pd
        
        # Convert dict to DataFrame
        df = pd.DataFrame([student_data])
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Extract features in correct order (must match training)
        feature_columns = [
            'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
            'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm',
            'quiz_avg', 'assignment_avg', 'overall_avg',
            'quiz_trend', 'assignment_trend',
            'quiz_std', 'assignment_std',
            'attendance_score_interaction',
            'high_performer', 'low_performer'
        ]
        
        # Handle NaNs (same as training)
        df_engineered = df_engineered.fillna(0)
        
        X = df_engineered[feature_columns].values
        
        # Scale features if scaler is available (it should be now)
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def predict(self, student_data):
        """
        Make predictions for a student.
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with predictions
        """
        # Prepare input
        input_data = self.prepare_input(student_data)
        
        # Make predictions
        pass_fail = self.pass_fail_model.predict(input_data)[0]
        pass_fail_proba = self.pass_fail_model.predict_proba(input_data)[0]
        
        final_score = self.final_score_model.predict(input_data)[0]
        
        support = self.support_model.predict(input_data)[0]
        support_proba = self.support_model.predict_proba(input_data)[0]
        
        # Calculate confidence scores
        pass_confidence = max(pass_fail_proba) * 100
        support_confidence = max(support_proba) * 100
        
        # Estimate regression confidence (using simple variance of trees if available, else heuristic)
        # For RF Regressor, we can get predictions from individual estimators to estimate uncertainty
        if hasattr(self.final_score_model, 'estimators_'):
            predictions = [tree.predict(input_data)[0] for tree in self.final_score_model.estimators_]
            confidence_interval_val = 2.0 * np.std(predictions)
        else:
            confidence_interval_val = 5.0 # Fallback
            
        # Get feature importance (from final score model as representative)
        feature_importance = {}
        if hasattr(self.final_score_model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.final_score_model.feature_importances_))
        
        # Prepare results
        results = {
            'pass_fail': {
                'prediction': 'Pass' if pass_fail == 1 else 'Fail',
                'confidence': round(pass_confidence, 2),
                'probability_pass': round(pass_fail_proba[1] * 100, 2),
                'probability_fail': round(pass_fail_proba[0] * 100, 2)
            },
            'final_exam_score': {
                'predicted_score': round(final_score, 2),
                'confidence_interval': round(confidence_interval_val, 2),
                'grade': self._get_grade(final_score),
                'min_score': round(max(0, final_score - confidence_interval_val), 2),
                'max_score': round(min(100, final_score + confidence_interval_val), 2)
            },
            'support_needed': {
                'prediction': 'Yes' if support == 1 else 'No',
                'confidence': round(support_confidence, 2),
                'recommendation': self._get_recommendation(support, final_score)
            },
            'overall_assessment': self._get_overall_assessment(
                pass_fail, final_score, support
            ),
            'features_used': student_data,
            'feature_importance': feature_importance,
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

# Create global predictor instance
predictor = None

def get_predictor():
    """Get or create predictor instance."""
    global predictor
    if predictor is None:
        predictor = StudentPerformancePredictor()
    return predictor
