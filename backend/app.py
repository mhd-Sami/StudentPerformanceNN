"""
Flask API Server
Provides REST API endpoints for student performance prediction.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.predictor import get_predictor
from models.predictor_nn import get_nn_predictor
from utils.validators import validate_student_data, sanitize_student_data

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize predictors
try:
    predictor = get_predictor()
except Exception as e:
    print(f"Warning: Random Forest predictor error: {e}")
    predictor = None

try:
    nn_predictor = get_nn_predictor()
except Exception as e:
    print(f"Warning: Neural Network predictor not available: {e}")
    nn_predictor = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict student performance.
    
    Expected JSON body:
    {
        "subject_id": "default" (optional),
        "attendance": float (0-100),
        "quiz1": float,
        "quiz2": float,
        "quiz3": float,
        "quiz4": float,
        "assignment1": float,
        "assignment2": float,
        "assignment3": float,
        "assignment4": float,
        "midterm": float
    }
    
    Note: For custom subjects, quiz/assignment/midterm values are obtained marks,
    not percentages. They will be normalized based on the subject configuration.
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide student data in JSON format'
            }), 400
        
        # Get model type (default: random_forest)
        model_type = data.get('model_type', 'random_forest')
        
        # Select appropriate predictor
        if model_type == 'neural_network':
            if nn_predictor is None:
                return jsonify({
                    'error': 'Neural Network model not available',
                    'message': 'Please train NN models first or use Random Forest'
                }), 500
            selected_predictor = nn_predictor
        else:  # default to random_forest
            if predictor is None:
                return jsonify({
                    'error': 'Random Forest model not loaded',
                    'message': 'Please train models first'
                }), 500
            selected_predictor = predictor
        
        # Get subject configuration ID (default if not provided)
        subject_id = data.get('subject_id', 'default')
        
        # If using custom subject, normalize scores
        if subject_id != 'default':
            from utils.subject_config import get_subject_config_manager
            
            try:
                manager = get_subject_config_manager()
                # Get subject config
                subject_config = manager.get_config(subject_id)
                
                # Normalize obtained marks to percentages
                normalized_data = manager.normalize_scores(data, subject_id)
                
                # Fill missing assessments with student's average
                normalized_data = manager.fill_missing_with_average(normalized_data, subject_config)
                
            except ValueError as e:
                return jsonify({
                    'error': 'Score normalization error',
                    'message': str(e)
                }), 400
        else:
            # Default: assume data is already in percentages
            normalized_data = data
            subject_config = None
        
        # Validate normalized data
        is_valid, error_message = validate_student_data(normalized_data)
        if not is_valid:
            return jsonify({
                'error': 'Validation error',
                'message': error_message
            }), 400
        
        # Sanitize data
        sanitized_data = sanitize_student_data(normalized_data)
        
        # Make prediction
        predictions = selected_predictor.predict(sanitized_data)
        
        # Model Comparison Logic
        comparison = None
        debug_info = []
        try:
            debug_info.append(f"Model Type: {model_type}")
            debug_info.append(f"RF Predictor exists: {predictor is not None}")
            debug_info.append(f"NN Predictor exists: {nn_predictor is not None}")
            
            if model_type == 'neural_network' and predictor is not None:
                # Compare with Random Forest
                debug_info.append("Entering NN vs RF comparison")
                rf_preds = predictor.predict(sanitized_data)
                comparison = {
                    'other_model_name': 'Random Forest',
                    'other_score': float(rf_preds['final_exam_score']['predicted_score']),
                    'other_pass_fail': rf_preds['pass_fail']['prediction'],
                    'agreement': bool(predictions['pass_fail']['prediction'] == rf_preds['pass_fail']['prediction'])
                }
            elif (model_type == 'random_forest' or model_type is None) and nn_predictor is not None:
                # Compare with Neural Network
                debug_info.append("Entering RF vs NN comparison")
                nn_preds = nn_predictor.predict(sanitized_data)
                comparison = {
                    'other_model_name': 'Neural Network',
                    'other_score': float(nn_preds['final_exam_score']['predicted_score']),
                    'other_pass_fail': nn_preds['pass_fail']['prediction'],
                    'agreement': bool(predictions['pass_fail']['prediction'] == nn_preds['pass_fail']['prediction'])
                }
            else:
                 debug_info.append("No matching condition for comparison")
                 
            debug_info.append(f"Comparison Result: {comparison}")
            
        except Exception as e:
            error_msg = f"Comparison failed: {str(e)}"
            print(error_msg)
            debug_info.append(error_msg)
            import traceback
            traceback.print_exc()
        
        # Return results
        response = {
            'success': True,
            'predictions': predictions,
            'comparison': comparison,
            'normalized_scores': sanitized_data,
            'debug_info': debug_info
        }
        
        # Include original marks and subject info if custom subject
        if subject_id != 'default' and subject_config:
            response['subject'] = {
                'id': subject_id,
                'name': subject_config['name'],
                'description': subject_config.get('description', '')
            }
            response['obtained_marks'] = {
                k: v for k, v in data.items() 
                if k != 'subject_id' and k != 'attendance'
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get list of available prediction models."""
    models = []
    
    # Check Random Forest availability
    if predictor is not None:
        models.append({
            'id': 'random_forest',
            'name': 'Random Forest',
            'description': 'Traditional ML using ensemble decision trees',
            'algorithm': 'RandomForestClassifier & RandomForestRegressor',
            'available': True
        })
    
    # Check Neural Network availability
    if nn_predictor is not None:
        models.append({
            'id': 'neural_network',
            'name': 'Neural Network',
            'description': 'Deep learning model with feature engineering',
            'algorithm': 'Multi-Layer Perceptron (MLP)',
            'available': True
        })
    
    return jsonify({
        'success': True,
        'models': models,
        'default': 'random_forest' if predictor is not None else ('neural_network' if nn_predictor is not None else None)
    })

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information."""
    return jsonify({
        'name': 'Student Performance Prediction API',
        'version': '1.0.0',
        'description': 'AI-based student performance prediction system',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/predict': 'Make predictions (POST)',
            '/api/info': 'API information',
            '/api/subject/create': 'Create subject configuration (POST)',
            '/api/subject/list': 'List all configurations (GET)',
            '/api/subject/<id>': 'Get specific configuration (GET)',
            '/api/subject/<id>/delete': 'Delete configuration (DELETE)'
        },
        'input_fields': [
            'attendance', 'quiz1', 'quiz2', 'quiz3', 'quiz4',
            'assignment1', 'assignment2', 'assignment3', 'assignment4', 'midterm'
        ],
        'predictions': [
            'pass_fail',
            'final_exam_score',
            'support_needed'
        ]
    })

@app.route('/api/subject/create', methods=['POST'])
def create_subject():
    """Create a new subject configuration."""
    try:
        from utils.subject_config import get_subject_config_manager
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide subject configuration data'
            }), 400
        
        manager = get_subject_config_manager()
        config = manager.create_config(data)
        
        return jsonify({
            'success': True,
            'message': 'Subject configuration created successfully',
            'config': config
        })
    
    except ValueError as e:
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Failed to create configuration',
            'message': str(e)
        }), 500

@app.route('/api/subject/list', methods=['GET'])
def list_subjects():
    """List all subject configurations."""
    try:
        from utils.subject_config import get_subject_config_manager
        
        manager = get_subject_config_manager()
        configs = manager.list_configs()
        
        # Convert to list format with summaries
        config_list = []
        for config_id, config in configs.items():
            summary = manager.get_config_summary(config_id)
            config_list.append(summary)
        
        return jsonify({
            'success': True,
            'configs': config_list
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to list configurations',
            'message': str(e)
        }), 500

@app.route('/api/subject/<config_id>', methods=['GET'])
def get_subject(config_id):
    """Get a specific subject configuration."""
    try:
        from utils.subject_config import get_subject_config_manager
        
        manager = get_subject_config_manager()
        config = manager.get_config(config_id)
        
        if not config:
            return jsonify({
                'error': 'Configuration not found',
                'message': f'No configuration found with ID: {config_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'config': config
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to get configuration',
            'message': str(e)
        }), 500

@app.route('/api/subject/<config_id>/delete', methods=['DELETE'])
def delete_subject(config_id):
    """Delete a subject configuration."""
    try:
        from utils.subject_config import get_subject_config_manager
        
        manager = get_subject_config_manager()
        success = manager.delete_config(config_id)
        
        if not success:
            return jsonify({
                'error': 'Configuration not found',
                'message': f'No configuration found with ID: {config_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'message': 'Configuration deleted successfully'
        })
    
    except ValueError as e:
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete configuration',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("Student Performance Prediction API")
    print("="*60)
    print("\nStarting server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /api/health  - Health check")
    print("  POST /api/predict - Make predictions")
    print("  GET  /api/info    - API information")
    print("\n" + "="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
