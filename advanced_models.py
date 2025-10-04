"""
Advanced Models Module
Supports XGBoost, LightGBM, CatBoost, and Keras MLP with safe imports
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Safe imports with fallback
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False
KERAS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from tensorflow import keras
    from keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    pass


def get_available_models():
    """Return dictionary of available advanced models"""
    models = {}
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = 'XGBoost'
    
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = 'LightGBM'
    
    if CATBOOST_AVAILABLE:
        models['catboost'] = 'CatBoost'
    
    if KERAS_AVAILABLE:
        models['keras_mlp'] = 'Keras MLP'
    
    return models


def get_xgboost_model(params=None):
    """Get XGBoost classifier with default or custom parameters"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    default_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    if params:
        default_params.update(params)
    
    return xgb.XGBClassifier(**default_params)


def get_lightgbm_model(params=None):
    """Get LightGBM classifier with default or custom parameters"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    default_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'verbose': -1
    }
    
    if params:
        default_params.update(params)
    
    return lgb.LGBMClassifier(**default_params)


def get_catboost_model(params=None):
    """Get CatBoost classifier with default or custom parameters"""
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not installed. Run: pip install catboost")
    
    default_params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'random_state': 42,
        'verbose': False,
        'allow_writing_files': False
    }
    
    if params:
        default_params.update(params)
    
    return CatBoostClassifier(**default_params)


class KerasMLP(BaseEstimator, ClassifierMixin):
    """
    Keras Multi-Layer Perceptron wrapper for scikit-learn compatibility
    """
    
    def __init__(self, hidden_layers=(64, 32), learning_rate=0.001, 
                 epochs=50, batch_size=32, random_state=42):
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras not installed. Run: pip install tensorflow")
        
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        self.n_features_in_ = None
    
    def _build_model(self, n_features, n_classes):
        """Build Keras sequential model"""
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(n_features,)))
        
        # Hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y):
        """Fit the Keras model"""
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Build model
        self.model = self._build_model(self.n_features_in_, len(self.classes_))
        
        # Train with validation split
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if len(self.classes_) == 2:
            predictions = (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(self.model.predict(X, verbose=0), axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if len(self.classes_) == 2:
            proba = self.model.predict(X, verbose=0)
            return np.column_stack([1 - proba, proba])
        else:
            return self.model.predict(X, verbose=0)


def get_keras_mlp_model(params=None):
    """Get Keras MLP classifier"""
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras not installed. Run: pip install tensorflow")
    
    default_params = {
        'hidden_layers': (64, 32),
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'random_state': 42
    }
    
    if params:
        # Convert hidden_layer_sizes to tuple if needed
        if 'hidden_layer_sizes' in params:
            params['hidden_layers'] = tuple(params.pop('hidden_layer_sizes'))
        default_params.update(params)
    
    return KerasMLP(**default_params)


def get_advanced_model(model_type, params=None):
    """
    Factory function to get advanced models
    
    Args:
        model_type: One of 'xgboost', 'lightgbm', 'catboost', 'keras_mlp'
        params: Dictionary of model parameters
    
    Returns:
        Model instance
    """
    model_getters = {
        'xgboost': get_xgboost_model,
        'lightgbm': get_lightgbm_model,
        'catboost': get_catboost_model,
        'keras_mlp': get_keras_mlp_model
    }
    
    if model_type not in model_getters:
        raise ValueError(f"Unknown advanced model type: {model_type}")
    
    return model_getters[model_type](params)


def get_model_param_grid(model_type):
    """
    Get hyperparameter grid for tuning
    
    Args:
        model_type: Model type string
    
    Returns:
        Dictionary of parameter ranges for GridSearchCV/RandomizedSearchCV
    """
    param_grids = {
        'xgboost': {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'min_child_weight': [1, 3, 5]
        },
        'lightgbm': {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'num_leaves': [31, 50, 70]
        },
        'catboost': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'depth': [4, 6, 8]
        },
        'keras_mlp': {
            'hidden_layers': [(32,), (64, 32), (128, 64)],
            'learning_rate': [0.001, 0.01],
            'epochs': [30, 50],
            'batch_size': [16, 32]
        }
    }
    
    return param_grids.get(model_type, {})