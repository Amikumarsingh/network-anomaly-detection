import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Any
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self, window_size_seconds: int = 30):
        self.window_size = window_size_seconds
        self.log_buffer = deque(maxlen=1000)  # Keep last 1000 logs
        
    def add_log(self, log: Dict[str, Any]):
        """Add new log to buffer"""
        # Convert timestamp string to datetime if needed
        if isinstance(log['timestamp'], str):
            log['timestamp_dt'] = datetime.fromisoformat(log['timestamp'])
        else:
            log['timestamp_dt'] = log['timestamp']
            log['timestamp'] = log['timestamp'].isoformat()
        self.log_buffer.append(log)
    
    def extract_features(self) -> Dict[str, float]:
        """Extract features from current window of logs"""
        if len(self.log_buffer) < 5:  # Need minimum logs
            return self._get_default_features()
        
        # Get logs within time window
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.window_size)
        
        window_logs = [
            log for log in self.log_buffer 
            if log['timestamp_dt'] >= window_start
        ]
        
        if not window_logs:
            return self._get_default_features()
        
        df = pd.DataFrame(window_logs)
        
        features = {
            # Latency features
            'avg_latency': df['latency_ms'].mean(),
            'latency_variance': df['latency_ms'].var(),
            'max_latency': df['latency_ms'].max(),
            'latency_95th': df['latency_ms'].quantile(0.95),
            
            # Packet loss features
            'avg_packet_loss': df['packet_loss_pct'].mean(),
            'max_packet_loss': df['packet_loss_pct'].max(),
            'packet_loss_variance': df['packet_loss_pct'].var(),
            
            # Retry features
            'avg_retry_count': df['retry_count'].mean(),
            'max_retry_count': df['retry_count'].max(),
            'retry_rate': (df['retry_count'] > 0).mean(),
            
            # Error features
            'error_rate': (df['error_code'] != 200).mean(),
            'server_error_rate': (df['error_code'] >= 500).mean(),
            
            # Traffic features
            'avg_traffic': df['traffic_volume_mb'].mean(),
            'traffic_variance': df['traffic_volume_mb'].var(),
            'traffic_delta': df['traffic_volume_mb'].max() - df['traffic_volume_mb'].min(),
            
            # Connection features
            'avg_connection_resets': df['connection_resets'].mean(),
            'total_connection_resets': df['connection_resets'].sum(),
            
            # Temporal features
            'log_frequency': len(window_logs) / self.window_size,
            'unique_devices': df['device_id'].nunique(),
        }
        
        # Handle NaN values
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
                
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data"""
        return {
            'avg_latency': 50.0, 'latency_variance': 0.0, 'max_latency': 50.0, 'latency_95th': 50.0,
            'avg_packet_loss': 0.5, 'max_packet_loss': 0.5, 'packet_loss_variance': 0.0,
            'avg_retry_count': 0.2, 'max_retry_count': 0, 'retry_rate': 0.0,
            'error_rate': 0.05, 'server_error_rate': 0.02,
            'avg_traffic': 10.0, 'traffic_variance': 0.0, 'traffic_delta': 0.0,
            'avg_connection_resets': 0.1, 'total_connection_resets': 0,
            'log_frequency': 1.0, 'unique_devices': 1
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return list(self._get_default_features().keys())