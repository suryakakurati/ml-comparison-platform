"""
ML Model Comparison Platform - Flask Backend
Production-grade machine learning model testing and comparison system
"""

from flask import Flask, render_template, request, jsonify, session
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import traceback

# ML Libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from preprocessing import DataPreprocessor
from advanced_models import (
    get_available_models, get_advanced_model, get_model_param_grid,
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE, KERAS_AVAILABLE
)
from optuna_tuner import OptunaOptimizer
from mlflow_tracker import MLflowTracker
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
os.makedirs('experiments/optuna_studies', exist_ok=True)
os.makedirs('mlruns', exist_ok=True)

# Global storage for experiment results
experiment_results = []

# Initialize MLflow tracker
mlflow_tracker = MLflowTracker()


def allowed_file(filename):
    """Validate file extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_dataframe(df):
    """Comprehensive dataframe validation"""
    if df is None or df.empty:
        raise ValueError("Dataset is empty")
    
    if df.shape[0] < 10:
        raise ValueError("Dataset must have at least 10 rows")
    
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + target)")
    
    return True


def get_traditional_param_grid(model_type):
    """Get parameter grid for traditional sklearn models"""
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],  # Only l2 for compatibility
            'max_iter': [1000]
        },
        'decision_tree': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7]
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'neural_network': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
    }
    
    return param_grids.get(model_type, {})


def preprocess_data(df, target_column, test_size=0.2, random_state=42, preprocessing_config=None):
    """
    Robust data preprocessing with configurable options using DataPreprocessor
    """
    try:
        # Create preprocessor with config
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Run preprocessing pipeline
        result = preprocessor.preprocess(df, target_column, test_size, random_state)
        
        return result
    
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")


def get_model(model_type, params=None):
    """
    Factory function to create ML models with default or custom parameters
    Supports both traditional and advanced models
    """
    # Check if it's an advanced model
    advanced_models = get_available_models()
    if model_type in advanced_models:
        return get_advanced_model(model_type, params)
    
    # Traditional models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svm': SVC(random_state=42),
        'neural_network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = models[model_type]
    
    if params:
        model.set_params(**params)
    
    return model


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Train model and compute comprehensive metrics
    """
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
            'test_accuracy': float(accuracy_score(y_test, y_pred_test)),
            'precision': float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
        
        return metrics
    
    except Exception as e:
        raise Exception(f"Training error: {str(e)}")


def generate_confusion_matrix_plot(cm, model_name):
    """Generate confusion matrix visualization"""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        print(f"Plot generation error: {str(e)}")
        return None


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and dataset preview"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and validate dataset
        df = pd.read_csv(filepath)
        validate_dataframe(df)
        
        # Store filepath in session
        session['dataset_path'] = filepath
        
        # Return dataset info
        return jsonify({
            'success': True,
            'filename': filename,
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'column_names': df.columns.tolist(),
            'preview': df.head(5).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """Train a single model with specified parameters and optional tuning"""
    try:
        data = request.json
        
        # Validate session
        if 'dataset_path' not in session:
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        # Load dataset
        df = pd.read_csv(session['dataset_path'])
        
        # Extract parameters
        target_column = data.get('target_column')
        model_type = data.get('model_type')
        test_size = float(data.get('test_size', 0.2))
        
        # Get preprocessing config
        preprocessing_config = {
            'missing_strategy': data.get('missing_strategy', 'mean'),
            'encoding_strategy': data.get('encoding_strategy', 'auto'),
            'scaling_strategy': data.get('scaling_strategy', 'standard')
        }
        
        # Get tuning configuration
        tuning_method = data.get('tuning_method', 'none')  # 'none', 'grid', 'random', 'optuna'
        enable_mlflow = data.get('enable_mlflow', True)
        cv_folds = int(data.get('cv_folds', 5))
        
        # Validate inputs
        if not target_column or not model_type:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Preprocess data with config
        processed = preprocess_data(df, target_column, test_size, preprocessing_config=preprocessing_config)
        
        # Get model parameters
        model_params = data.get('params', {})
        
        # Variable to store final model and metrics
        final_model = None
        metrics = None
        cv_scores = None
        tuning_results = None
        
        # Apply hyperparameter tuning based on method
        if tuning_method == 'optuna':
            # Optuna optimization
            try:
                model_factory = lambda params: get_model(model_type, params)
                optimizer = OptunaOptimizer(
                    model_factory,
                    processed['X_train'],
                    processed['y_train'],
                    n_trials=int(data.get('n_trials', 30)),
                    cv=cv_folds
                )
                
                optuna_result = optimizer.optimize(model_type)
                model_params = optuna_result['best_params']
                
                # Save Optuna study
                study_path = optimizer.save_study()
                
                tuning_results = {
                    'method': 'optuna',
                    'best_params': optuna_result['best_params'],
                    'best_score': optuna_result['best_score'],
                    'n_trials': optuna_result['n_trials'],
                    'top_trials': optimizer.get_top_trials(5),
                    'study_path': study_path
                }
            except Exception as e:
                return jsonify({'error': f'Optuna optimization failed: {str(e)}'}), 500
        
        elif tuning_method in ['grid', 'random']:
            # GridSearchCV or RandomizedSearchCV
            try:
                # Get parameter grid
                if model_type in get_available_models():
                    param_grid = get_model_param_grid(model_type)
                else:
                    param_grid = get_traditional_param_grid(model_type)
                
                if not param_grid:
                    return jsonify({'error': f'No parameter grid defined for {model_type}'}), 400
                
                base_model = get_model(model_type)
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=cv_folds,
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                else:  # random
                    search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        cv=cv_folds,
                        scoring='f1_weighted',
                        n_iter=20,
                        n_jobs=-1,
                        random_state=42
                    )
                
                search.fit(processed['X_train'], processed['y_train'])
                model_params = search.best_params_
                cv_scores = [search.best_score_]
                
                tuning_results = {
                    'method': tuning_method,
                    'best_params': search.best_params_,
                    'best_score': float(search.best_score_),
                    'cv_folds': cv_folds
                }
            except Exception as e:
                return jsonify({'error': f'{tuning_method.title()} search failed: {str(e)}'}), 500
        
        # Train final model with best parameters
        final_model = get_model(model_type, model_params)
        
        # Perform cross-validation if not already done
        if cv_scores is None and cv_folds > 1:
            cv_scores = cross_val_score(
                final_model,
                processed['X_train'],
                processed['y_train'],
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=-1
            )
        
        # Train and evaluate
        metrics = train_and_evaluate(
            final_model,
            processed['X_train'],
            processed['X_test'],
            processed['y_train'],
            processed['y_test']
        )
        
        # Generate confusion matrix plot
        cm_plot = generate_confusion_matrix_plot(
            np.array(metrics['confusion_matrix']),
            model_type
        )
        
        # Log to MLflow if enabled
        mlflow_run_id = None
        if enable_mlflow:
            try:
                mlflow_run_id = mlflow_tracker.log_experiment(
                    model=final_model,
                    model_type=model_type,
                    params=model_params,
                    metrics=metrics,
                    confusion_matrix=np.array(metrics['confusion_matrix']),
                    preprocessing_info=processed.get('preprocessing_report', {}),
                    cv_scores=cv_scores.tolist() if cv_scores is not None else None,
                    tuning_method=tuning_method
                )
            except Exception as e:
                print(f"MLflow logging failed: {str(e)}")
        
        # Store result with all information
        result = {
            'model_type': model_type,
            'params': model_params,
            'metrics': metrics,
            'confusion_matrix_plot': cm_plot,
            'preprocessing': processed.get('preprocessing_report', {}),
            'tuning': tuning_results,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'mlflow_run_id': mlflow_run_id,
            'timestamp': datetime.now().isoformat()
        }
        
        experiment_results.append(result)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        print(f"Training error: {traceback.format_exc()}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


@app.route('/auto-optimize', methods=['POST'])
def auto_optimize():
    """Automatically find best model and hyperparameters"""
    try:
        data = request.json
        
        # Validate session
        if 'dataset_path' not in session:
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        # Load dataset
        df = pd.read_csv(session['dataset_path'])
        
        # Extract parameters
        target_column = data.get('target_column')
        test_size = float(data.get('test_size', 0.2))
        search_type = data.get('search_type', 'grid')
        
        # Get preprocessing config
        preprocessing_config = {
            'missing_strategy': data.get('missing_strategy', 'mean'),
            'encoding_strategy': data.get('encoding_strategy', 'auto'),
            'scaling_strategy': data.get('scaling_strategy', 'standard')
        }
        
        if not target_column:
            return jsonify({'error': 'Target column required'}), 400
        
        # Preprocess data with config
        processed = preprocess_data(df, target_column, test_size, preprocessing_config=preprocessing_config)
        
        # Define models and parameter grids
        model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5]
                }
            },
            'svm': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        results = []
        
        # Test each model configuration
        for model_name, config in model_configs.items():
            try:
                # Perform grid/random search
                if search_type == 'grid':
                    search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=3,
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                else:
                    search = RandomizedSearchCV(
                        config['model'],
                        config['params'],
                        cv=3,
                        scoring='f1_weighted',
                        n_iter=10,
                        n_jobs=-1,
                        random_state=42
                    )
                
                # Fit search
                search.fit(processed['X_train'], processed['y_train'])
                
                # Evaluate best model
                best_model = search.best_estimator_
                metrics = train_and_evaluate(
                    best_model,
                    processed['X_train'],
                    processed['X_test'],
                    processed['y_train'],
                    processed['y_test']
                )
                
                # Generate confusion matrix
                cm_plot = generate_confusion_matrix_plot(
                    np.array(metrics['confusion_matrix']),
                    model_name
                )
                
                result = {
                    'model_type': model_name,
                    'params': search.best_params_,
                    'metrics': metrics,
                    'confusion_matrix_plot': cm_plot,
                    'cv_score': float(search.best_score_),
                    'preprocessing': processed.get('preprocessing_report', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                experiment_results.append(result)
            
            except Exception as model_error:
                print(f"Error with {model_name}: {str(model_error)}")
                continue
        
        if not results:
            return jsonify({'error': 'All models failed to train'}), 500
        
        # Sort by F1 score
        results.sort(key=lambda x: x['metrics']['f1_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'results': results,
            'best_model': results[0]
        })
    
    except Exception as e:
        print(f"Auto-optimize error: {traceback.format_exc()}")
        return jsonify({'error': f'Auto-optimization failed: {str(e)}'}), 500


@app.route('/results', methods=['GET'])
def get_results():
    """Retrieve all experiment results"""
    return jsonify({
        'success': True,
        'results': experiment_results
    })


@app.route('/clear-results', methods=['POST'])
def clear_results():
    """Clear all experiment results"""
    global experiment_results
    experiment_results = []
    return jsonify({'success': True})


@app.route('/available-models', methods=['GET'])
def available_models():
    """Get list of all available models"""
    traditional_models = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'neural_network': 'Neural Network (MLP)'
    }
    
    advanced_models = get_available_models()
    
    return jsonify({
        'success': True,
        'traditional_models': traditional_models,
        'advanced_models': advanced_models,
        'all_models': {**traditional_models, **advanced_models}
    })


@app.route('/mlflow-info', methods=['GET'])
def mlflow_info():
    """Get MLflow tracking information"""
    try:
        recent_runs = mlflow_tracker.get_experiment_runs(n=10)
        
        return jsonify({
            'success': True,
            'tracking_uri': mlflow_tracker.tracking_uri,
            'experiment_name': mlflow_tracker.experiment_name,
            'experiment_id': mlflow_tracker.experiment_id,
            'ui_command': mlflow_tracker.start_ui(),
            'recent_runs_count': len(recent_runs)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get MLflow info: {str(e)}'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error occurred'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render assigns a port automatically
    app.run(debug=True, host='0.0.0.0', port=port)