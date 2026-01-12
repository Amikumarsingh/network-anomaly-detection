import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple

class IsolationForestModel:
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def train(self, features_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Train the isolation forest model"""
        if len(features_list) < 50:
            raise ValueError("Need at least 50 samples for training")
        
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Calculate training metrics
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        metrics = {
            'training_samples': len(features_list),
            'anomaly_rate': (predictions == -1).mean(),
            'avg_anomaly_score': -scores.mean(),  # Negative for intuitive scoring
            'score_std': scores.std()
        }
        
        return metrics
    
    def predict(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict if features represent an anomaly"""
        if not self.is_trained:
            return False, 0.0
        
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(df)
        
        # Get prediction and score
        prediction = self.model.predict(X_scaled)[0]
        score = -self.model.decision_function(X_scaled)[0]  # Negative for intuitive scoring
        
        is_anomaly = prediction == -1
        return is_anomaly, float(score)
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']