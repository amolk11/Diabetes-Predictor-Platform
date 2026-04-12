"""Metrics computation utilities"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score
)

def compute_all_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute all evaluation metrics"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    
    return metrics

def print_metrics_report(y_true, y_pred):
    """Print comprehensive metrics report"""
    metrics = compute_all_metrics(y_true, y_pred)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    print("\nCONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred)
    print(f"TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
    print(f"FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")
    
    print("\nCLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred))
