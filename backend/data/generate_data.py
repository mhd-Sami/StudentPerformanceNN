"""
Synthetic Student Data Generator
Generates realistic training data for student performance prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_data(n_samples=5000):
    """
    Generate synthetic student performance data with diverse profiles.
    
    Args:
        n_samples: Number of student records to generate (default 5000)
        
    Returns:
        DataFrame with student performance data
    """
    data = []
    
    # Define student profiles (weights sum to 1.0)
    profiles = [
        {'name': 'High Achiever', 'weight': 0.30, 'beta_a': 12, 'beta_b': 2, 'noise': 0.05},
        {'name': 'Average',       'weight': 0.50, 'beta_a': 5,  'beta_b': 5, 'noise': 0.10},
        {'name': 'Struggling',    'weight': 0.20, 'beta_a': 2,  'beta_b': 5, 'noise': 0.15}
    ]
    
    for _ in range(n_samples):
        # Select profile
        profile = np.random.choice(profiles, p=[p['weight'] for p in profiles])
        
        # Generate varied base performance
        base = np.random.beta(profile['beta_a'], profile['beta_b'])
        
        # Add profile-specific noise
        noise = np.random.normal(0, profile['noise'])
        perf = np.clip(base + noise, 0, 1)
        
        # 1. Attendance (High achievers attend more, but some variance)
        att_base = perf * 100
        # Add "smart slacker" outlier chance (high perf, low attendance)
        if np.random.random() < 0.05 and perf > 0.8:
            attendance = np.random.uniform(50, 80)
        else:
            attendance = np.clip(att_base + np.random.normal(0, 8), 40, 100)
            
        # 2. Quizzes (Standard variance)
        quiz_func = lambda: np.clip(perf * 100 + np.random.normal(0, 10), 0, 100)
        quizzes = [quiz_func() for _ in range(4)]
        
        # 3. Assignments (Usually higher than quizzes)
        # Add "hard worker" chance (low perf, high assignment)
        assign_boost = 5
        if np.random.random() < 0.05 and perf < 0.6:
            assign_boost = 15 # Tries hard on homework
            
        assign_func = lambda: np.clip(perf * 100 + assign_boost + np.random.normal(0, 8), 0, 100)
        assignments = [assign_func() for _ in range(4)]
        
        # 4. Midterm (Harder exam)
        midterm = np.clip(perf * 100 - 2 + np.random.normal(0, 12), 0, 100)
        
        # 5. Final Exam (Target) 
        # Correlated with overall but independent variance
        final_exam = np.clip(perf * 100 + np.random.normal(0, 10), 0, 100)
        
        # --- Derived Metrics ---
        # Pass/Fail (Strict cut-off)
        pass_fail = 1 if final_exam >= 50 else 0
        
        # Support Needed (Heuristc)
        avg_score = np.mean(quizzes + assignments + [midterm])
        needs_support = 1 if (avg_score < 60 or attendance < 65 or (midterm < 50 and avg_score < 70)) else 0
        
        data.append({
            'attendance': round(attendance, 2),
            'quiz1': round(quizzes[0], 2),
            'quiz2': round(quizzes[1], 2),
            'quiz3': round(quizzes[2], 2),
            'quiz4': round(quizzes[3], 2),
            'assignment1': round(assignments[0], 2),
            'assignment2': round(assignments[1], 2),
            'assignment3': round(assignments[2], 2),
            'assignment4': round(assignments[3], 2),
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
    df = generate_student_data(n_samples=5000)
    
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
