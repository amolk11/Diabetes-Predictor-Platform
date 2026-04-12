#!/usr/bin/env python3
"""Model Evaluation Script with MLflow Integration"""

import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report, 
    matthews_corrcoef, cohen_kappa_score, roc_curve, auc
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with MLflow integration"""
    
    def __init__(self, model_path='models/trained/best_model.pkl', 
                 scaler_path='models/scaler/scaler.pkl',
                 data_path='data/diabetes.csv',
                 metrics_output='models/metrics/evaluation_report.json'):
        """Initialize evaluator"""
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.df = pd.read_csv(data_path)
        self.metrics_output = Path(metrics_output)
        self.metrics = {}
        
    @staticmethod
    def _load_model(model_path):
        """Load trained model"""
        if not Path(model_path).exists():
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model file: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    
    @staticmethod
    def _load_scaler(scaler_path):
        """Load data scaler"""
        if not Path(scaler_path).exists():
            logger.warning(f"Scaler not found at {scaler_path}")
            return None
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
        return scaler
    
    def evaluate(self) -> Dict:
        """Run full evaluation"""
        logger.info("=" * 70)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 70)
        
        # Prepare data
        target = 'Outcome' if 'Outcome' in self.df.columns else self.df.columns[-1]
        X = self.df.drop(columns=[target]).values
        y = self.df[target].values
        
        # Scale if scaler available
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Generate predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        self._calculate_metrics(y, y_pred, y_pred_proba)
        self._print_metrics()
        self._print_confusion_matrix(y, y_pred)
        self._print_classification_report(y, y_pred)
        
        # Save evaluation report
        self._save_evaluation_report()
        
        return self.metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        self.metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'matthews_cc': float(matthews_corrcoef(y_true, y_pred)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
            'specificity': float(self._calculate_specificity(y_true, y_pred)),
            'sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
            'evaluation_timestamp': datetime.now().isoformat(),
            'samples_evaluated': int(len(y_true))
        }
        
        # Add confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
    
    @staticmethod
    def _calculate_specificity(y_true, y_pred):
        """Calculate specificity"""
        cm = confusion_matrix(y_true, y_pred)
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity
    
    def _print_metrics(self):
        """Print metrics nicely formatted"""
        print("\n" + "=" * 70)
        print("CLASSIFICATION METRICS")
        print("=" * 70)
        
        print(f"Accuracy:        {self.metrics['accuracy']:.4f}")
        print(f"Precision:       {self.metrics['precision']:.4f}")
        print(f"Recall:          {self.metrics['recall']:.4f}")
        print(f"Specificity:     {self.metrics['specificity']:.4f}")
        print(f"F1 Score:        {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:         {self.metrics['roc_auc']:.4f}")
        print(f"Matthews CC:     {self.metrics['matthews_cc']:.4f}")
        print(f"Cohen's Kappa:   {self.metrics['cohen_kappa']:.4f}")
        print("=" * 70)
    
    def _print_confusion_matrix(self, y_true, y_pred):
        """Print confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        print(f"                 Predicted Negative  Predicted Positive")
        print(f"Actual Negative   {cm[0,0]:>6}              {cm[0,1]:>6}")
        print(f"Actual Positive   {cm[1,0]:>6}              {cm[1,1]:>6}")
        print("=" * 70)
    
    def _print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
        print("=" * 70)
    
    def _save_evaluation_report(self):
        """Save evaluation report as JSON"""
        report_data = {
            'metrics': self.metrics,
            'evaluation_date': datetime.now().isoformat(),
            'model_location': 'models/trained/best_model.pkl'
        }
        
        self.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.metrics_output, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Evaluation report saved to {self.metrics_output}")
    
    def log_to_mlflow(self, run_name='evaluation'):
        """Log evaluation results to MLflow"""
        mlflow.set_experiment("diabetes-prediction")
        
        with mlflow.start_run(run_name=run_name):
            # Log metrics
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log confusion matrix as artifacts
            cm_dict = self.metrics.get('confusion_matrix', {})
            cm_path = Path('models/metrics/confusion_matrix.json')
            with open(cm_path, 'w') as f:
                json.dump(cm_dict, f, indent=2)
            
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(self.metrics_output))
            
            logger.info("Evaluation results logged to MLflow")


def main():
    """Main evaluation function"""
    try:
        evaluator = ModelEvaluator(
            model_path='models/trained/best_model.pkl',
            scaler_path='models/scaler/scaler.pkl',
            data_path='data/diabetes.csv'
        )
        
        metrics = evaluator.evaluate()
        
        # Log to MLflow
        evaluator.log_to_mlflow()
        
        logger.info("\n✓ Evaluation complete!")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    

if __name__ == "__main__":
    main()
