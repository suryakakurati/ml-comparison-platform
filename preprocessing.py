"""
Preprocessing Pipeline Module
Handles data cleaning, encoding, and scaling operations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Comprehensive data preprocessing with configurable options
    """
    
    def __init__(self, config=None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config (dict): Preprocessing options
                - missing_strategy: 'mean', 'median', 'mode', 'drop'
                - encoding_strategy: 'label', 'onehot', 'auto'
                - scaling_strategy: 'standard', 'minmax', 'none'
        """
        self.config = config or {
            'missing_strategy': 'mean',
            'encoding_strategy': 'auto',
            'scaling_strategy': 'standard'
        }
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_names = []
        self.report = {}
        
    def handle_missing_values(self, df, strategy=None):
        """Handle missing values in dataframe"""
        strategy = strategy or self.config.get('missing_strategy', 'mean')
        missing_before = int(df.isnull().sum().sum())
        
        if missing_before == 0:
            self.report['missing_values'] = {'strategy': strategy, 'values_imputed': 0}
            return df
        
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
            self.report['missing_values'] = {
                'strategy': 'drop',
                'rows_dropped': missing_before
            }
        
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
            
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
            
            self.report['missing_values'] = {
                'strategy': 'mean',
                'values_imputed': missing_before
            }
        
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
            
            self.report['missing_values'] = {
                'strategy': 'median',
                'values_imputed': missing_before
            }
        
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
            
            self.report['missing_values'] = {
                'strategy': 'mode',
                'values_imputed': missing_before
            }
        
        return df
    
    def encode_categorical(self, X, strategy=None):
        """Encode categorical variables"""
        strategy = strategy or self.config.get('encoding_strategy', 'auto')
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) == 0:
            self.report['encoding'] = {'strategy': strategy, 'columns_encoded': 0}
            return X
        
        X = X.copy()
        
        if strategy == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            self.report['encoding'] = {
                'strategy': 'label',
                'columns_encoded': int(len(categorical_cols)),
                'columns': categorical_cols
            }
        
        elif strategy == 'onehot':
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            self.report['encoding'] = {
                'strategy': 'onehot',
                'columns_encoded': int(len(categorical_cols)),
                'columns': categorical_cols,
                'new_features': int(len(X.columns))
            }
        
        elif strategy == 'auto':
            onehot_cols = []
            label_cols = []
            
            for col in categorical_cols:
                unique_count = X[col].nunique()
                if unique_count <= 10:
                    onehot_cols.append(col)
                else:
                    label_cols.append(col)
            
            if onehot_cols:
                X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
            
            for col in label_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            self.report['encoding'] = {
                'strategy': 'auto',
                'onehot_columns': onehot_cols,
                'label_columns': label_cols,
                'total_encoded': int(len(categorical_cols))
            }
        
        return X
    
    def scale_features(self, X_train, X_test, strategy=None):
        """Scale numerical features"""
        strategy = strategy or self.config.get('scaling_strategy', 'standard')
        
        if strategy == 'none':
            self.report['scaling'] = {'strategy': 'none'}
            return X_train, X_test
        
        if strategy == 'standard':
            self.scaler = StandardScaler()
        elif strategy == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.report['scaling'] = {
            'strategy': strategy,
            'scaler_type': type(self.scaler).__name__
        }
        
        return X_train_scaled, X_test_scaled
    
    def encode_target(self, y):
        """Encode target variable if categorical"""
        if y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
            
            self.report['target_encoding'] = {
                'encoded': True,
                'classes': [str(c) for c in self.target_encoder.classes_],
                'num_classes': int(len(self.target_encoder.classes_))
            }
            
            return y_encoded
        
        self.report['target_encoding'] = {'encoded': False}
        return y.values
    
    def validate_target(self, y):
        """Validate target distribution"""
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        if min_count < 2:
            classes_with_one = [str(unique[i]) for i, c in enumerate(counts) if c < 2]
            raise ValueError(
                f"Target has classes with only 1 sample: {', '.join(classes_with_one)}. "
                f"Each class needs at least 2 samples."
            )
        
        self.report['target_validation'] = {
            'classes': int(len(unique)),
            'min_samples': int(min_count),
            'max_samples': int(counts.max())
        }
    
    def preprocess(self, df, target_column, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input dataframe
            target_column: Target column name
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            dict with X_train, X_test, y_train, y_test, and metadata
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Step 3: Encode categorical features
        X = self.encode_categorical(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Step 4: Encode target
        y = self.encode_target(y)
        
        # Step 5: Validate target
        self.validate_target(y)
        
        # Step 6: Train-test split
        unique, counts = np.unique(y, return_counts=True)
        use_stratify = counts.min() >= 2
        
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Step 7: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Add split info to report
        self.report['split'] = {
            'test_size': float(test_size),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'stratified': bool(use_stratify)
        }
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'preprocessing_report': self.report
        }