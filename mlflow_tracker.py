"""
MLflow Experiment Tracking Module
Provides local experiment logging and tracking
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import mlflow.keras
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MLflowTracker:
    """
    Wrapper for MLflow experiment tracking
    """
    
    def __init__(self, tracking_uri='mlruns', experiment_name='ml_comparison'):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: URI for MLflow tracking (local directory)
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(f"file:./{tracking_uri}")
        
        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
    
    def log_experiment(self, model, model_type, params, metrics, 
                      confusion_matrix=None, preprocessing_info=None,
                      cv_scores=None, tuning_method=None):
        """
        Log a complete experiment to MLflow
        
        Args:
            model: Trained model instance
            model_type: String identifier for model type
            params: Dictionary of model parameters
            metrics: Dictionary of performance metrics
            confusion_matrix: Confusion matrix array
            preprocessing_info: Dictionary of preprocessing details
            cv_scores: Cross-validation scores
            tuning_method: Type of tuning used ('grid', 'random', 'optuna', 'none')
        
        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("tuning_method", tuning_method or "none")
            
            # Log model hyperparameters
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    mlflow.log_param(key, str(value))
            
            # Log metrics
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    try:
                        mlflow.log_metric(key, float(value))
                    except (ValueError, TypeError):
                        pass
            
            # Log cross-validation scores if available
            if cv_scores:
                mlflow.log_metric("cv_mean", float(np.mean(cv_scores)))
                mlflow.log_metric("cv_std", float(np.std(cv_scores)))
            
            # Log preprocessing info as JSON
            if preprocessing_info:
                mlflow.log_dict(preprocessing_info, "preprocessing_config.json")
            
            # Log confusion matrix as artifact
            if confusion_matrix is not None:
                cm_path = self._save_confusion_matrix(confusion_matrix, model_type)
                if cm_path:
                    mlflow.log_artifact(cm_path)
                    os.remove(cm_path)  # Clean up temp file
            
            # Log model
            try:
                self._log_model(model, model_type)
            except Exception as e:
                print(f"Could not log model: {e}")
            
            # Get run ID
            run_id = mlflow.active_run().info.run_id
            
            return run_id
    
    def _log_model(self, model, model_type):
        """Log model to MLflow with appropriate flavor"""
        model_info = {
            'xgboost': lambda m: mlflow.xgboost.log_model(m, "model"),
            'lightgbm': lambda m: mlflow.lightgbm.log_model(m, "model"),
            'catboost': lambda m: mlflow.catboost.log_model(m, "model"),
            'keras_mlp': lambda m: mlflow.keras.log_model(m.model, "model") if hasattr(m, 'model') else None
        }
        
        if model_type in model_info:
            model_info[model_type](model)
        else:
            mlflow.sklearn.log_model(model, "model")
    
    def _save_confusion_matrix(self, cm, model_name):
        """Save confusion matrix as image"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save to temp file
            temp_path = f"temp_cm_{model_name}.png"
            plt.savefig(temp_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            return temp_path
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
            return None
    
    def get_experiment_runs(self, n=10):
        """
        Get recent experiment runs
        
        Args:
            n: Number of recent runs to retrieve
        
        Returns:
            List of run dictionaries
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=n,
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            return []
        
        return runs.to_dict('records')
    
    def compare_runs(self, run_ids):
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
        
        Returns:
            DataFrame with comparison
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"run_id IN {tuple(run_ids)}"
        )
        
        return runs
    
    @staticmethod
    def start_ui(port=5001):
        """
        Instructions to start MLflow UI
        
        Args:
            port: Port number for MLflow UI
        
        Returns:
            Command string
        """
        return f"mlflow ui --backend-store-uri file://./mlruns --port {port}"