# Student Performance Prediction System

An AI-based student performance prediction system that uses machine learning to predict:
- **Pass/Fail Status** - Will the student pass before taking the final exam?
- **Final Exam Score** - Predicted performance in the final exam (0-100)
- **Support Need** - Whether the student needs additional academic support

## 🎯 Features

- **Machine Learning Models**: Three Random Forest models for comprehensive predictions
- **Real-time Predictions**: Instant analysis of student performance
- **Confidence Scores**: Probability-based predictions with confidence metrics
- **Modern UI**: Beautiful, responsive interface with dark mode and animations
- **RESTful API**: Flask-based backend with CORS support
- **Comprehensive Insights**: Detailed recommendations and overall assessment

## 📊 Input Features

The system analyzes the following student data:
- Attendance percentage (0-100%)
- Quiz scores: Quiz 1, 2, 3, 4 (0-100 each)
- Assignment scores: Assignment 1, 2, 3, 4 (0-100 each)
- Midterm exam score (0-100)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone or download the project**

2. **Navigate to the project directory**
   ```bash
   cd student-performance-prediction
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Setup and Training

1. **Generate training data**
   ```bash
   python backend/data/generate_data.py
   ```
   This creates a synthetic dataset of 1000 student records.

2. **Train the models**
   ```bash
   python backend/models/train_models.py
   ```
   This trains three ML models and saves them to `backend/models/saved_models/`.

### Running the Application

1. **Start the backend API**
   ```bash
   python backend/app.py
   ```
   The API will be available at `http://localhost:5000`

2. **Open the frontend**
   - Open `frontend/index.html` in your web browser
   - Or use a local server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
   - Then navigate to `http://localhost:8000`

3. **Make predictions**
   - Fill in the student information form
   - Click "Predict Performance"
   - View the comprehensive results and recommendations

## 📁 Project Structure

```
student-performance-prediction/
├── backend/
│   ├── app.py                      # Flask API server
│   ├── requirements.txt            # Python dependencies
│   ├── models/
│   │   ├── train_models.py         # Model training script
│   │   ├── predictor.py            # Prediction logic
│   │   └── saved_models/           # Trained models (generated)
│   ├── data/
│   │   ├── generate_data.py        # Synthetic data generator
│   │   └── training_data.csv       # Training dataset (generated)
│   └── utils/
│       └── validators.py           # Input validation
├── frontend/
│   ├── index.html                  # Main UI
│   ├── styles.css                  # Styling
│   └── script.js                   # Frontend logic
└── README.md                       # This file
```

## 🔌 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Health Check
```http
GET /api/health
```
Returns API status and model loading status.

#### Make Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "attendance": 85.0,
  "quiz1": 78.0,
  "quiz2": 82.0,
  "quiz3": 75.0,
  "quiz4": 88.0,
  "assignment1": 85.0,
  "assignment2": 90.0,
  "assignment3": 87.0,
  "assignment4": 92.0,
  "midterm": 80.0
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "pass_fail": {
      "prediction": "Pass",
      "confidence": 95.5,
      "probability_pass": 95.5,
      "probability_fail": 4.5
    },
    "final_exam_score": {
      "predicted_score": 82.5,
      "grade": "B"
    },
    "support_needed": {
      "prediction": "No",
      "confidence": 88.2,
      "recommendation": "Good performance. Continue current study habits."
    },
    "overall_assessment": "Strong Performance - Student is on track for excellent results."
  }
}
```

#### API Info
```http
GET /api/info
```
Returns API information and available endpoints.

## 🤖 Model Details

### Pass/Fail Classifier
- **Algorithm**: Random Forest Classifier
- **Purpose**: Predicts whether student will pass (≥50) or fail (<50)
- **Output**: Binary classification with probability scores

### Final Score Predictor
- **Algorithm**: Random Forest Regressor
- **Purpose**: Predicts numerical final exam score (0-100)
- **Output**: Predicted score and letter grade (A-F)

### Support Need Classifier
- **Algorithm**: Random Forest Classifier
- **Purpose**: Determines if student needs additional academic support
- **Criteria**: Based on average performance and attendance
- **Output**: Binary classification with recommendations

### Model Performance
After training on 1000 synthetic records:
- Pass/Fail Accuracy: ~85-90%
- Final Score RMSE: ~8-12 points
- Support Need Accuracy: ~85-90%

## 💡 Usage Tips

1. **Demo Data**: Press `Ctrl+D` on the form to auto-fill with sample data
2. **Validation**: All inputs must be between 0-100
3. **API Connection**: Ensure backend is running before using the frontend
4. **Model Training**: Re-train models with more data for better accuracy

## 🎨 UI Features

- **Dark Mode**: Modern dark theme with gradient accents
- **Glassmorphism**: Frosted glass effects on cards
- **Animations**: Smooth transitions and hover effects
- **Responsive**: Works on desktop, tablet, and mobile
- **Color Coding**: Visual indicators for pass/fail and support status

## 🔧 Customization

### Adjusting Model Parameters
Edit `backend/models/train_models.py`:
```python
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

### Changing Passing Grade
Edit `backend/models/predictor.py`:
```python
# Current: 50 is passing
pass_fail = 1 if final_exam >= 50 else 0
```

### Generating More Data
Edit `backend/data/generate_data.py`:
```python
df = generate_student_data(n_samples=5000)  # Increase sample size
```

## 🐛 Troubleshooting

### Models Not Found Error
```
Solution: Run python backend/models/train_models.py
```

### API Connection Failed
```
Solution: Ensure backend is running on port 5000
Check: http://localhost:5000/api/health
```

### CORS Error
```
Solution: flask-cors is installed and configured
Check: backend/requirements.txt includes flask-cors
```

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more features
- Improve model accuracy
- Enhance UI/UX
- Fix bugs

## 📧 Support

For issues or questions, please create an issue in the project repository.

---

**Built with ❤️ for Academic Success**
