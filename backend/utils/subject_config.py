"""
Subject Configuration Module
Manages subject configurations with custom total marks for assessments.
"""

import json
from pathlib import Path
from typing import Dict, Optional

class SubjectConfig:
    """Manages subject configurations."""
    
    def __init__(self):
        """Initialize subject configuration manager."""
        self.config_file = Path(__file__).parent.parent / 'data' / 'subject_configs.json'
        self.configs = self.load_all_configs()
    
    def load_all_configs(self) -> Dict:
        """Load all subject configurations from file."""
        if not self.config_file.exists():
            # Create default configuration
            default_configs = {
                "default": self.create_default_config()
            }
            self.save_all_configs(default_configs)
            return default_configs
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configs: {e}")
            return {"default": self.create_default_config()}
    
    def save_all_configs(self, configs: Dict):
        """Save all configurations to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def create_default_config(self) -> Dict:
        """Create default configuration (all out of 100)."""
        return {
            "id": "default",
            "name": "Default (Out of 100)",
            "description": "All assessments are out of 100 marks",
            "num_quizzes": 4,
            "num_assignments": 4,
            "has_midterm": True,
            "quiz1_total": 100,
            "quiz2_total": 100,
            "quiz3_total": 100,
            "quiz4_total": 100,
            "assignment1_total": 100,
            "assignment2_total": 100,
            "assignment3_total": 100,
            "assignment4_total": 100,
            "midterm_total": 100
        }
    
    def create_config(self, config_data: Dict) -> Dict:
        """
        Create a new subject configuration.
        
        Args:
            config_data: Dictionary with configuration details
            
        Returns:
            Created configuration
        """
        # Validate required fields
        required_fields = [
            'id', 'name', 'num_quizzes', 'num_assignments', 'has_midterm',
            'quiz1_total', 'quiz2_total', 'quiz3_total', 'quiz4_total',
            'assignment1_total', 'assignment2_total', 'assignment3_total', 
            'assignment4_total', 'midterm_total'
        ]
        
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate total marks are positive
        total_fields = [f for f in required_fields if f.endswith('_total')]
        for field in total_fields:
            if config_data[field] <= 0:
                raise ValueError(f"{field} must be positive")
        
        # Validate assessment counts
        if not 1 <= config_data['num_quizzes'] <= 4:
            raise ValueError("num_quizzes must be between 1 and 4")
        if not 1 <= config_data['num_assignments'] <= 4:
            raise ValueError("num_assignments must be between 1 and 4")
        
        # Add description if not provided
        if 'description' not in config_data:
            config_data['description'] = f"Custom configuration for {config_data['name']}"
        
        # Save configuration
        self.configs[config_data['id']] = config_data
        self.save_all_configs(self.configs)
        
        return config_data
    
    def get_config(self, config_id: str) -> Optional[Dict]:
        """Get a specific configuration by ID."""
        return self.configs.get(config_id)
    
    def list_configs(self) -> Dict:
        """List all available configurations."""
        return self.configs
    
    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration."""
        if config_id == "default":
            raise ValueError("Cannot delete default configuration")
        
        if config_id in self.configs:
            del self.configs[config_id]
            self.save_all_configs(self.configs)
            return True
        return False
    
    def normalize_scores(self, obtained_marks: Dict, config_id: str = "default") -> Dict:
        """
        Normalize obtained marks to percentages based on configuration.
        
        Args:
            obtained_marks: Dictionary with obtained marks
            config_id: Subject configuration ID
            
        Returns:
            Dictionary with normalized scores (percentages)
        """
        config = self.get_config(config_id)
        if not config:
            raise ValueError(f"Configuration not found: {config_id}")
        
        normalized = {}
        
        # Normalize each component
        components = [
            'quiz1', 'quiz2', 'quiz3', 'quiz4',
            'assignment1', 'assignment2', 'assignment3', 'assignment4',
            'midterm'
        ]
        
        for component in components:
            if component in obtained_marks:
                total_key = f"{component}_total"
                total_marks = config[total_key]
                obtained = obtained_marks[component]
                
                # Skip if obtained is None (will be filled later with average)
                if obtained is None:
                    continue
                
                # Validate obtained marks
                if obtained < 0:
                    raise ValueError(f"{component} cannot be negative")
                if obtained > total_marks:
                    raise ValueError(f"{component} ({obtained}) exceeds total marks ({total_marks})")
                
                # Calculate percentage
                normalized[component] = (obtained / total_marks) * 100
        
        # Add attendance (already a percentage)
        if 'attendance' in obtained_marks and obtained_marks['attendance'] is not None:
            normalized['attendance'] = obtained_marks['attendance']
        
        return normalized
    
    def get_active_assessments(self, config: Dict) -> list:
        """
        Get list of active assessment fields based on configuration.
        
        Args:
            config: Subject configuration
            
        Returns:
            List of active field names
        """
        active = ['attendance']  # Always include attendance
        
        # Add active quizzes
        for i in range(1, config['num_quizzes'] + 1):
            active.append(f'quiz{i}')
        
        # Add active assignments
        for i in range(1, config['num_assignments'] + 1):
            active.append(f'assignment{i}')
        
        # Add midterm if conducted
        if config['has_midterm']:
            active.append('midterm')
        
        return active
    
    def fill_missing_with_average(self, obtained_marks: Dict, config: Dict) -> Dict:
        """
        Fill missing assessments with student's average performance.
        
        Args:
            obtained_marks: Dictionary with obtained marks (normalized percentages)
            config: Subject configuration
            
        Returns:
            Complete dictionary with all 10 fields filled
        """
        # Get active assessments
        active = self.get_active_assessments(config)
        
        # Calculate average from conducted assessments (exclude attendance)
        conducted_scores = []
        for field in active:
            if field != 'attendance' and field in obtained_marks:
                conducted_scores.append(obtained_marks[field])
        
        # Calculate average (or use 50 as default if no scores)
        average_score = sum(conducted_scores) / len(conducted_scores) if conducted_scores else 50.0
        
        # Fill complete data
        complete_data = {}
        all_fields = [
            'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
            'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm'
        ]
        
        for field in all_fields:
            value = obtained_marks.get(field)
            if value is not None:
                complete_data[field] = value
            elif field == 'attendance':
                # If attendance missing or None, use 100 (assume full attendance)
                complete_data[field] = 100.0
            else:
                # Fill with average
                complete_data[field] = average_score
        
        return complete_data
    
    def get_config_summary(self, config_id: str) -> Dict:
        """Get a summary of a configuration for display."""
        config = self.get_config(config_id)
        if not config:
            return None
        
        return {
            'id': config['id'],
            'name': config['name'],
            'description': config.get('description', ''),
            'num_quizzes': config.get('num_quizzes', 4),
            'num_assignments': config.get('num_assignments', 4),
            'has_midterm': config.get('has_midterm', True),
            'total_marks': {
                'quizzes': [
                    config['quiz1_total'],
                    config['quiz2_total'],
                    config['quiz3_total'],
                    config['quiz4_total']
                ],
                'assignments': [
                    config['assignment1_total'],
                    config['assignment2_total'],
                    config['assignment3_total'],
                    config['assignment4_total']
                ],
                'midterm': config['midterm_total']
            }
        }

# Global instance
_subject_config_manager = None

def get_subject_config_manager():
    """Get or create subject configuration manager instance."""
    global _subject_config_manager
    if _subject_config_manager is None:
        _subject_config_manager = SubjectConfig()
    return _subject_config_manager
