"""
Synthetic Student Data Generator
Generates realistic training data for student performance prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_data(n_samples=1000):
    """
    Generate synthetic student performance data.
    
    Args:
        n_samples: Number of student records to generate
        
    Returns:
        DataFrame with student performance data
    """
    data = []
    
    for _ in range(n_samples):
        # Generate base performance level (0-1) that influences all metrics
        base_performance = np.random.beta(5, 2)  # Skewed towards higher performance
        
        # Add some randomness
        noise = np.random.normal(0, 0.15)
        performance_level = np.clip(base_performance + noise, 0, 1)
        
        # Attendance (correlated with performance)
        attendance = np.clip(
            performance_level * 100 + np.random.normal(0, 10),
            0, 100
        )
        
        # Quiz scores (4 quizzes)
        quiz_base = performance_level * 100
        quiz1 = np.clip(quiz_base + np.random.normal(0, 12), 0, 100)
        quiz2 = np.clip(quiz_base + np.random.normal(0, 12), 0, 100)
        quiz3 = np.clip(quiz_base + np.random.normal(0, 12), 0, 100)
        quiz4 = np.clip(quiz_base + np.random.normal(0, 12), 0, 100)
        
        # Assignment scores (4 assignments) - slightly higher than quizzes
        assignment_base = performance_level * 100 + 5
        assignment1 = np.clip(assignment_base + np.random.normal(0, 10), 0, 100)
        assignment2 = np.clip(assignment_base + np.random.normal(0, 10), 0, 100)
        assignment3 = np.clip(assignment_base + np.random.normal(0, 10), 0, 100)
        assignment4 = np.clip(assignment_base + np.random.normal(0, 10), 0, 100)
        
        # Midterm exam
        midterm = np.clip(
            performance_level * 100 + np.random.normal(0, 15),
            0, 100
        )
        
        # Final exam (target variable) - correlated with overall performance
        final_exam = np.clip(
            performance_level * 100 + np.random.normal(0, 12),
            0, 100
        )
        
        # Calculate average performance
        avg_performance = np.mean([
            attendance, quiz1, quiz2, quiz3, quiz4,
            assignment1, assignment2, assignment3, assignment4, midterm
        ])
        
        # Pass/Fail (passing grade is 50)
        pass_fail = 1 if final_exam >= 50 else 0
        
        # Needs support (if average performance < 60 or attendance < 70)
        needs_support = 1 if (avg_performance < 60 or attendance < 70) else 0
        
        data.append({
            'attendance': round(attendance, 2),
            'quiz1': round(quiz1, 2),
            'quiz2': round(quiz2, 2),
            'quiz3': round(quiz3, 2),
            'quiz4': round(quiz4, 2),
            'assignment1': round(assignment1, 2),
            'assignment2': round(assignment2, 2),
            'assignment3': round(assignment3, 2),
            'assignment4': round(assignment4, 2),
            'midterm': round(midterm, 2),
            'final_exam': round(final_exam, 2),
            'pass_fail': pass_fail,
            'needs_support': needs_support
        })
    
    return pd.DataFrame(data)

def main():
    """Generate and save training data."""
    print("Generating synthetic student data...")
    
    # Generate data
    df = generate_student_data(n_samples=50000)
    
    # Save to CSV
    output_path = Path(__file__).parent / 'training_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} student records")
    print(f"✓ Saved to: {output_path}")
    print(f"\nData Statistics:")
    print(f"  Pass Rate: {df['pass_fail'].mean() * 100:.1f}%")
    print(f"  Students Needing Support: {df['needs_support'].mean() * 100:.1f}%")
    print(f"  Average Final Exam Score: {df['final_exam'].mean():.2f}")
    print(f"  Average Attendance: {df['attendance'].mean():.2f}%")
    
    # Display sample records
    print(f"\nSample Records:")
    print(df.head())

if __name__ == "__main__":
    main()
