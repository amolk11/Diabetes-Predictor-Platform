"""Model prediction module"""
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Make predictions with trained model"""
    
    def __init__(self, model_path, scaler_path=None):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.scaler = None
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        logger.info("Model loaded for inference")
    
    def predict(self, X):
        """Predict on batch of samples"""
        if self.scaler:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, features):
        """Predict single sample"""
        features = np.array(features).reshape(1, -1)
        pred, proba = self.predict(features)
        return {
            'prediction': int(pred[0]),
            'probability': float(proba[0][1]),
            'confidence': float(max(proba[0]))
        }
