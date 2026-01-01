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
    
    def prepare_input(self, student_data):
        """
        Prepare input data for prediction.
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            numpy array ready for prediction
        """
        # Extract features in correct order
        features = [
            student_data['attendance'],
            student_data['quiz1'],
            student_data['quiz2'],
            student_data['quiz3'],
            student_data['quiz4'],
            student_data['assignment1'],
            student_data['assignment2'],
            student_data['assignment3'],
            student_data['assignment4'],
            student_data['midterm']
        ]
        
        return np.array([features])
    
    def predict(self, student_data):
        """
        Make predictions for a student.
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with predictions
        """
        # Prepare input
        X = self.prepare_input(student_data)
        
        # Make predictions
        pass_fail_pred = self.pass_fail_model.predict(X)[0]
        pass_fail_proba = self.pass_fail_model.predict_proba(X)[0]
        
        final_score_pred = self.final_score_model.predict(X)[0]
        
        support_pred = self.support_model.predict(X)[0]
        support_proba = self.support_model.predict_proba(X)[0]
        
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
            'features_used': student_data,
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
