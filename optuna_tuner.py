"""
Optuna Hyperparameter Optimization Module
Provides smart, efficient parameter tuning
"""

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
import json
import os
from datetime import datetime

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Wrapper for Optuna optimization
    """
    
    def __init__(self, model_factory, X_train, y_train, n_trials=50, cv=5):
        """
        Initialize Optuna optimizer
        
        Args:
            model_factory: Function that returns a model given parameters
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
        """
        self.model_factory = model_factory
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.cv = cv
        self.study = None
        self.best_params = None
        self.best_score = None
    
    def _objective_logistic(self, trial):
        """Objective function for Logistic Regression"""
        params = {
            'C': trial.suggest_float('C', 0.001, 10.0, log=True),
            'penalty': 'l2',  # Use l2 for compatibility with default solver
            'max_iter': 1000,
            'random_state': 42
        }
        
        model = self.model_factory(params)
        score = cross_val_score(model, self.X_train, self.y_train, 
                               cv=self.cv, scoring='f1_weighted', n_jobs=-1).mean()
        return score
    
    def _objective_random_forest(self, trial):
        """Objective function for Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
        model = self.model_factory(params)
        score = cross_val_score(model, self.X_train, self.y_train,
                               cv=self.cv, scoring='f1_weighted', n_jobs=-1).mean()
        return score
    
    def _objective_xgboost(self, trial):
        """Objective function for XGBoost"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        model = self.model_factory(params)
        score = cross_val_score(model, self.X_train, self.y_train,
                               cv=self.cv, scoring='f1_weighted', n_jobs=-1).mean()
        return score
    
    def _objective_lightgbm(self, trial):
        """Objective function for LightGBM"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        model = self.model_factory(params)
        score = cross_val_score(model, self.X_train, self.y_train,
                               cv=self.cv, scoring='f1_weighted', n_jobs=-1).mean()
        return score
    
    def _objective_catboost(self, trial):
        """Objective function for CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
        }
        
        model = self.model_factory(params)
        score = cross_val_score(model, self.X_train, self.y_train,
                               cv=self.cv, scoring='f1_weighted', n_jobs=-1).mean()
        return score
    
    def _objective_gradient_boosting(self, trial):
        """Objective function for Gradient Boosting"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
        
        model = self.model_factory(params)
        score = cross_val_score(model, self.X_train, self.y_train,
                               cv=self.cv, scoring='f1_weighted', n_jobs=-1).mean()
        return score
    
    def optimize(self, model_type):
        """
        Run Optuna optimization
        
        Args:
            model_type: Type of model to optimize
        
        Returns:
            Dictionary with best parameters and score
        """
        objective_map = {
            'logistic_regression': self._objective_logistic,
            'random_forest': self._objective_random_forest,
            'gradient_boosting': self._objective_gradient_boosting,
            'xgboost': self._objective_xgboost,
            'lightgbm': self._objective_lightgbm,
            'catboost': self._objective_catboost
        }
        
        if model_type not in objective_map:
            raise ValueError(f"Optuna optimization not supported for {model_type}")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_optimization'
        )
        
        # Optimize
        self.study.optimize(
            objective_map[model_type],
            n_trials=self.n_trials,
            show_progress_bar=False
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'n_trials': self.n_trials,
            'study_name': self.study.study_name
        }
    
    def get_top_trials(self, n=5):
        """
        Get top N trials from the study
        
        Args:
            n: Number of top trials to return
        
        Returns:
            List of dictionaries with trial information
        """
        if self.study is None:
            return []
        
        trials = sorted(self.study.trials, key=lambda t: t.value, reverse=True)[:n]
        
        return [
            {
                'trial_number': trial.number,
                'params': trial.params,
                'score': float(trial.value)
            }
            for trial in trials
        ]
    
    def save_study(self, output_dir='experiments/optuna_studies'):
        """
        Save Optuna study results to JSON
        
        Args:
            output_dir: Directory to save study results
        """
        if self.study is None:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.study.study_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        study_data = {
            'study_name': self.study.study_name,
            'timestamp': timestamp,
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'n_trials': self.n_trials,
            'top_trials': self.get_top_trials(5)
        }
        
        with open(filepath, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        return filepath