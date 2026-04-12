"""Feature engineering module"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create and engineer features"""
    
    @staticmethod
    def create_polynomial_features(X, degree=2):
        """Generate polynomial features"""
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        logger.info(f"Polynomial features: {X.shape[1]} -> {X_poly.shape[1]}")
        return X_poly, poly
    
    @staticmethod
    def create_interaction_features(X, feature_indices=None):
        """Create interaction features between specified columns"""
        X_interaction = X.copy()
        
        if feature_indices is None:
            feature_indices = [(i, j) for i in range(X.shape[1]) 
                             for j in range(i+1, X.shape[1])]
        
        for i, j in feature_indices:
            interaction = X[:, i] * X[:, j]
            X_interaction = np.column_stack([X_interaction, interaction])
        
        logger.info(f"Interaction features created: {X.shape[1]} -> {X_interaction.shape[1]}")
        return X_interaction
    
    @staticmethod
    def create_statistical_features(X):
        """Create statistical features"""
        X_stats = np.column_stack([
            X,
            np.mean(X, axis=1),  # row mean
            np.std(X, axis=1),   # row std
            np.max(X, axis=1),   # row max
            np.min(X, axis=1),   # row min
        ])
        logger.info(f"Statistical features created")
        return X_stats
