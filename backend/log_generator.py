import time
import random
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np

class NetworkLogGenerator:
    def __init__(self):
        self.device_ids = [f"device_{i:03d}" for i in range(1, 11)]
        self.error_codes = [200, 404, 500, 502, 503, 504, 408, 429]
        self.anomaly_probability = 0.05
        self.current_anomaly_duration = 0
        self.anomaly_type = None
        
    def generate_normal_log(self) -> Dict[str, Any]:
        """Generate normal network behavior log"""
        return {
            "timestamp": datetime.now().isoformat(),
            "device_id": random.choice(self.device_ids),
            "latency_ms": max(10, np.random.normal(50, 15)),
            "packet_loss_pct": max(0, np.random.normal(0.5, 0.3)),
            "retry_count": np.random.poisson(0.2),
            "error_code": random.choices([200, 404, 500], weights=[0.95, 0.03, 0.02])[0],
            "traffic_volume_mb": max(0.1, np.random.normal(10, 3)),
            "connection_resets": np.random.poisson(0.1)
        }
    
    def generate_anomaly_log(self, anomaly_type: str) -> Dict[str, Any]:
        """Generate anomalous network behavior"""
        log = self.generate_normal_log()
        
        if anomaly_type == "latency_spike":
            log["latency_ms"] = np.random.normal(300, 50)
            log["retry_count"] = np.random.poisson(3)
            
        elif anomaly_type == "packet_loss":
            log["packet_loss_pct"] = np.random.normal(15, 5)
            log["retry_count"] = np.random.poisson(5)
            
        elif anomaly_type == "error_burst":
            log["error_code"] = random.choice([500, 502, 503, 504])
            log["retry_count"] = np.random.poisson(8)
            
        elif anomaly_type == "traffic_anomaly":
            log["traffic_volume_mb"] = np.random.normal(100, 20)
            log["connection_resets"] = np.random.poisson(5)
            
        return log
    
    def generate_log(self) -> Dict[str, Any]:
        """Generate a single log entry with potential anomalies"""
        # Manage anomaly duration
        if self.current_anomaly_duration > 0:
            self.current_anomaly_duration -= 1
            log = self.generate_anomaly_log(self.anomaly_type)
            log["is_anomaly"] = True
            return log
        
        # Start new anomaly
        if random.random() < self.anomaly_probability:
            self.anomaly_type = random.choice([
                "latency_spike", "packet_loss", "error_burst", "traffic_anomaly"
            ])
            self.current_anomaly_duration = random.randint(5, 15)  # 5-15 seconds
            log = self.generate_anomaly_log(self.anomaly_type)
            log["is_anomaly"] = True
            return log
        
        # Normal log
        log = self.generate_normal_log()
        log["is_anomaly"] = False
        return log