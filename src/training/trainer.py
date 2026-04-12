"""Model training module"""
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import logging
import pickle

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate models"""
    
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
    
    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train model and compute metrics"""
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=self.cv_strategy, scoring='f1'
        )
        
        # Training
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        logger.info(f"{model_name}: F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        return model, metrics
    
    @staticmethod
    def save_model(model, path):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved: {path}")
    
    @staticmethod
    def load_model(path):
        """Load trained model"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {path}")
        return model
