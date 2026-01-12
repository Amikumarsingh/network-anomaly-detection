"""
Test script to verify the anomaly detection system components
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from log_generator import NetworkLogGenerator
from feature_engineering import FeatureEngineer
from models.isolation_forest import IsolationForestModel
import time

def test_log_generation():
    """Test log generator"""
    print("Testing log generation...")
    generator = NetworkLogGenerator()
    
    # Generate some logs
    logs = []
    for i in range(10):
        log = generator.generate_log()
        logs.append(log)
        print(f"Log {i+1}: Latency={log['latency_ms']:.1f}ms, Anomaly={log['is_anomaly']}")
    
    anomaly_count = sum(1 for log in logs if log['is_anomaly'])
    print(f"Generated {len(logs)} logs, {anomaly_count} anomalies")
    return logs

def test_feature_engineering(logs):
    """Test feature engineering"""
    print("\nTesting feature engineering...")
    engineer = FeatureEngineer(window_size_seconds=30)
    
    # Add logs to buffer
    for log in logs:
        engineer.add_log(log)
    
    # Extract features
    features = engineer.extract_features()
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    return features

def test_model_training():
    """Test model training"""
    print("\nTesting model training...")
    
    # Generate training data
    generator = NetworkLogGenerator()
    engineer = FeatureEngineer()
    training_features = []
    
    print("Generating training data...")
    for i in range(100):
        log = generator.generate_normal_log()  # Only normal logs for training
        log['is_anomaly'] = False
        engineer.add_log(log)
        features = engineer.extract_features()
        training_features.append(features)
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    model = IsolationForestModel()
    metrics = model.train(training_features)
    
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test prediction
    test_features = training_features[0]
    is_anomaly, score = model.predict(test_features)
    print(f"\nTest prediction: Anomaly={is_anomaly}, Score={score:.3f}")
    
    return model

def main():
    """Run all tests"""
    print("=== Network Anomaly Detection System Test ===\n")
    
    try:
        # Test components
        logs = test_log_generation()
        features = test_feature_engineering(logs)
        model = test_model_training()
        
        print("\n=== All Tests Passed! ===")
        print("\nTo run the full system:")
        print("1. cd backend")
        print("2. python main.py")
        print("3. Open frontend/index.html in browser")
        print("4. Click 'Train Models' then 'Start Detection'")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()