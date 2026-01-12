import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import os

class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int):
        super(AutoencoderNet, self).__init__()
        hidden_dim = max(8, input_dim // 2)
        bottleneck_dim = max(4, input_dim // 4)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderModel:
    def __init__(self, threshold_percentile: float = 95):
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        self.feature_names = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, features_list: List[Dict[str, float]], epochs: int = 100) -> Dict[str, float]:
        """Train the autoencoder model"""
        if len(features_list) < 50:
            raise ValueError("Need at least 50 samples for training")
        
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        input_dim = len(self.feature_names)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Initialize model
        self.model = AutoencoderNet(input_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Calculate reconstruction errors for threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.threshold = torch.quantile(reconstruction_errors, self.threshold_percentile / 100.0).item()
        
        self.is_trained = True
        
        # Calculate training metrics
        anomaly_predictions = reconstruction_errors > self.threshold
        
        metrics = {
            'training_samples': len(features_list),
            'final_loss': losses[-1],
            'threshold': self.threshold,
            'anomaly_rate': anomaly_predictions.float().mean().item(),
            'avg_reconstruction_error': reconstruction_errors.mean().item()
        }
        
        return metrics
    
    def predict(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict if features represent an anomaly"""
        if not self.is_trained:
            return False, 0.0
        
        # Convert to tensor with correct feature order
        df = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(df)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get reconstruction error
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2).item()
        
        is_anomaly = reconstruction_error > self.threshold
        return is_anomaly, reconstruction_error
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'input_dim': len(self.feature_names)
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        input_dim = checkpoint['input_dim']
        self.model = AutoencoderNet(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']
        self.feature_names = checkpoint['feature_names']
        self.is_trained = True