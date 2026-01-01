import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from report_generator import generate_student_report

# Mock data with STRINGS to simulate the issue
mock_data = {
    "predictions": {
        "pass_fail": {"prediction": "Pass", "confidence": "95"},
        "final_exam_score": {"predicted_score": "75.5", "confidence_interval": "±5"},
        "needs_support": {"prediction": "No"}
    },
    "features": {
        "attendance": "85",
        "quiz1": "70",
        "quiz4": "80", 
        "midterm": "75"
    },
    "comparison": {
        "agreement": True,
        "other_model_name": "RF",
        "other_score": "74.2"
    }
}

try:
    print("Testing PDF Generation with String Inputs...")
    pdf = generate_student_report(mock_data)
    print(f"Success! PDF generated with size: {len(pdf)} bytes")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
