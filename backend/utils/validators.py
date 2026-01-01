"""
Input Validation Module
Validates student data before prediction.
"""

def validate_student_data(data):
    """
    Validate student input data.
    
    Args:
        data: Dictionary with student information
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
        'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm'
    ]
    
    # Check all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate attendance (0-100)
    try:
        attendance = float(data['attendance'])
        if not 0 <= attendance <= 100:
            return False, "Attendance must be between 0 and 100"
    except (ValueError, TypeError):
        return False, "Attendance must be a valid number"
    
    # Validate all scores (0-100)
    score_fields = [
        'quiz1', 'quiz2', 'quiz3', 'quiz4',
        'assignment1', 'assignment2', 'assignment3', 'assignment4',
        'midterm'
    ]
    
    for field in score_fields:
        try:
            score = float(data[field])
            if not 0 <= score <= 100:
                return False, f"{field} must be between 0 and 100"
        except (ValueError, TypeError):
            return False, f"{field} must be a valid number"
    
    return True, None

def sanitize_student_data(data):
    """
    Sanitize and convert student data to proper types.
    
    Args:
        data: Dictionary with student information
        
    Returns:
        Sanitized dictionary
    """
    sanitized = {}
    
    fields = [
        'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
        'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm'
    ]
    
    for field in fields:
        sanitized[field] = float(data[field])
    
    return sanitized
