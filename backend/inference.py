import asyncio
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json

from log_generator import NetworkLogGenerator
from feature_engineering import FeatureEngineer
from models.isolation_forest import IsolationForestModel
from models.autoencoder import AutoencoderModel

class InferenceEngine:
    def __init__(self):
        self.log_generator = NetworkLogGenerator()
        self.feature_engineer = FeatureEngineer(window_size_seconds=30)
        
        # Models
        self.isolation_forest = IsolationForestModel()
        self.autoencoder = AutoencoderModel()
        self.current_model = "isolation_forest"
        
        # State
        self.is_running = False
        self.training_data = []
        self.inference_results = []
        self.callbacks = []
        
        # Metrics
        self.total_logs = 0
        self.total_anomalies = 0
        self.detection_latencies = []
        
    def add_callback(self, callback: Callable):
        """Add callback for real-time updates"""
        self.callbacks.append(callback)
    
    def notify_callbacks(self, data: Dict[str, Any]):
        """Notify all callbacks with new data"""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Callback error: {e}")
    
    async def collect_training_data(self, duration_seconds: int = 120):
        """Collect training data from normal logs"""
        print(f"Collecting training data for {duration_seconds} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Generate mostly normal logs for training
            log = self.log_generator.generate_normal_log()
            log["is_anomaly"] = False
            
            self.feature_engineer.add_log(log)
            features = self.feature_engineer.extract_features()
            self.training_data.append(features)
            
            await asyncio.sleep(0.5)  # 2 logs per second
        
        print(f"Collected {len(self.training_data)} training samples")
    
    def train_models(self) -> Dict[str, Any]:
        """Train all available models"""
        if len(self.training_data) < 50:
            raise ValueError("Insufficient training data")
        
        results = {}
        
        # Train Isolation Forest
        try:
            if_metrics = self.isolation_forest.train(self.training_data)
            results["isolation_forest"] = {
                "status": "success",
                "metrics": if_metrics
            }
            print("Isolation Forest trained successfully")
        except Exception as e:
            results["isolation_forest"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Train Autoencoder
        try:
            ae_metrics = self.autoencoder.train(self.training_data, epochs=50)
            results["autoencoder"] = {
                "status": "success", 
                "metrics": ae_metrics
            }
            print("Autoencoder trained successfully")
        except Exception as e:
            results["autoencoder"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def set_model(self, model_name: str):
        """Switch active model"""
        if model_name in ["isolation_forest", "autoencoder"]:
            self.current_model = model_name
            print(f"Switched to {model_name}")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def predict_anomaly(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run inference with current model"""
        start_time = time.time()
        
        if self.current_model == "isolation_forest":
            is_anomaly, score = self.isolation_forest.predict(features)
        elif self.current_model == "autoencoder":
            is_anomaly, score = self.autoencoder.predict(features)
        else:
            return {"error": "No model selected"}
        
        detection_latency = (time.time() - start_time) * 1000  # ms
        self.detection_latencies.append(detection_latency)
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "model_used": self.current_model,
            "detection_latency_ms": detection_latency
        }
    
    async def start_inference(self):
        """Start real-time inference loop"""
        self.is_running = True
        print("Starting real-time inference...")
        
        while self.is_running:
            try:
                # Generate log
                log = self.log_generator.generate_log()
                self.total_logs += 1
                
                # Extract features
                self.feature_engineer.add_log(log)
                features = self.feature_engineer.extract_features()
                
                # Run inference
                prediction = self.predict_anomaly(features)
                
                # Track metrics
                if prediction.get("is_anomaly", False):
                    self.total_anomalies += 1
                
                # Prepare result
                result = {
                    "timestamp": log["timestamp"],
                    "log": log,
                    "features": features,
                    "prediction": prediction,
                    "ground_truth": log.get("is_anomaly", False)
                }
                
                self.inference_results.append(result)
                if len(self.inference_results) > 1000:  # Keep last 1000 results
                    self.inference_results.pop(0)
                
                # Notify callbacks
                self.notify_callbacks(result)
                
                await asyncio.sleep(1)  # 1 log per second
                
            except Exception as e:
                print(f"Inference error: {e}")
                await asyncio.sleep(1)
    
    def stop_inference(self):
        """Stop real-time inference"""
        self.is_running = False
        print("Stopped inference")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current evaluation metrics"""
        if not self.inference_results:
            return {"error": "No inference results available"}
        
        # Calculate metrics from recent results
        recent_results = self.inference_results[-100:]  # Last 100 results
        
        predictions = [r["prediction"]["is_anomaly"] for r in recent_results]
        ground_truth = [r["ground_truth"] for r in recent_results]
        
        # Basic metrics
        anomaly_rate = sum(predictions) / len(predictions) if predictions else 0
        true_anomaly_rate = sum(ground_truth) / len(ground_truth) if ground_truth else 0
        
        # Approximate precision/recall
        tp = sum(p and g for p, g in zip(predictions, ground_truth))
        fp = sum(p and not g for p, g in zip(predictions, ground_truth))
        fn = sum(not p and g for p, g in zip(predictions, ground_truth))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        avg_latency = sum(self.detection_latencies[-100:]) / len(self.detection_latencies[-100:]) if self.detection_latencies else 0
        
        return {
            "total_logs_processed": self.total_logs,
            "total_anomalies_detected": self.total_anomalies,
            "current_anomaly_rate": anomaly_rate,
            "true_anomaly_rate": true_anomaly_rate,
            "precision": precision,
            "recall": recall,
            "avg_detection_latency_ms": avg_latency,
            "current_model": self.current_model,
            "training_samples": len(self.training_data)
        }