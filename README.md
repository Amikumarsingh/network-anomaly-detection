# Real-Time Network Anomaly Detection System

A production-ready web application that demonstrates real-time anomaly detection on network/modem logs using machine learning. This system showcases end-to-end ML pipeline design, streaming data processing, and real-time inference capabilities.

## 🏗️ System Architecture

```
Log Generator → Feature Engineering → ML Models → Real-Time Inference → Web Dashboard
     ↓               ↓                   ↓              ↓                ↓
Network Logs    Sliding Windows    Isolation Forest   WebSocket      Live Charts
Simulation      Feature Extraction    Autoencoder     Updates        Anomaly Alerts
```

## 🔧 Tech Stack

**Backend:**
- FastAPI (REST APIs + WebSockets)
- Python 3.8+
- Scikit-learn (Isolation Forest)
- PyTorch (Autoencoder)
- Pandas/NumPy (Data processing)

**Frontend:**
- React (Dashboard)
- Chart.js (Real-time visualization)
- WebSocket (Live updates)

**Storage:**
- In-memory processing
- Model persistence with joblib/PyTorch

## 🚀 Quick Start

### 1. Setup Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8001`

### 2. Setup Frontend

Open `frontend/index.html` in your browser or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

### 3. Using the System

1. **Train Models**: Click "Train Models" to collect 60 seconds of training data
2. **Start Detection**: Click "Start Detection" to begin real-time anomaly detection
3. **Monitor**: Watch live charts and log streams for anomalies
4. **Switch Models**: Use dropdown to switch between Isolation Forest and Autoencoder

## 📊 Features Implemented

### Log Generation & Simulation
- **Realistic Network Logs**: Latency, packet loss, retry counts, error codes, traffic volume
- **Anomaly Simulation**: Latency spikes, packet loss bursts, error storms, traffic anomalies
- **Continuous Generation**: 1 log per second with 5% anomaly probability

### Feature Engineering
- **Sliding Time Windows**: 30-second windows for feature extraction
- **Key Features**:
  - Average/variance/95th percentile latency
  - Packet loss rates and variance
  - Retry rates and error frequencies
  - Traffic deltas and connection resets
  - Temporal features (log frequency, device diversity)

### Machine Learning Models

#### Isolation Forest
- **Approach**: Unsupervised outlier detection using isolation trees
- **Training**: On normal network behavior patterns
- **Output**: Anomaly score based on isolation difficulty
- **Strengths**: Fast, interpretable, works well with mixed data types

#### Autoencoder
- **Approach**: Neural network reconstruction-based anomaly detection
- **Architecture**: Encoder-decoder with bottleneck layer
- **Training**: Learn to reconstruct normal patterns
- **Output**: Reconstruction error as anomaly score
- **Strengths**: Captures complex non-linear patterns

### Real-Time Inference
- **Streaming Processing**: Live log ingestion and feature extraction
- **Model Switching**: Runtime model selection via API
- **Low Latency**: Sub-millisecond detection times
- **WebSocket Updates**: Real-time dashboard updates

### Evaluation & Metrics
- **Anomaly Rate**: Current detection rate
- **Precision/Recall**: Approximate using synthetic ground truth
- **Detection Latency**: Time from log to prediction
- **Model Comparison**: Side-by-side performance analysis

## 🎯 Production Considerations

### Scalability
- **Horizontal Scaling**: Stateless API design allows multiple instances
- **Streaming**: Can integrate with Kafka/Redis for high-throughput scenarios
- **Model Serving**: Models can be containerized and deployed separately

### Monitoring
- **Health Checks**: API endpoints for system status
- **Metrics Collection**: Built-in performance tracking
- **Alerting**: WebSocket-based real-time notifications

### Security
- **Input Validation**: Pydantic models for API validation
- **CORS Configuration**: Configurable cross-origin policies
- **Model Integrity**: Checksum validation for saved models

## 📈 Model Comparison

| Model | Training Time | Inference Speed | Memory Usage | Interpretability |
|-------|---------------|-----------------|--------------|------------------|
| Isolation Forest | ~2 seconds | <1ms | Low | High |
| Autoencoder | ~30 seconds | <5ms | Medium | Medium |

### When to Use Each Model

**Isolation Forest:**
- Fast deployment needed
- Interpretable results required
- Mixed data types
- Limited training data

**Autoencoder:**
- Complex pattern detection needed
- Sufficient training data available
- Can tolerate longer training times
- Non-linear relationships expected

## 🔍 Use Cases

### ISP/Telecom Applications
- **Network Performance Monitoring**: Detect service degradation
- **Capacity Planning**: Identify unusual traffic patterns
- **SLA Monitoring**: Alert on performance threshold breaches
- **Predictive Maintenance**: Early detection of equipment issues

### Security Applications
- **DDoS Detection**: Identify traffic anomalies
- **Intrusion Detection**: Unusual connection patterns
- **Fraud Detection**: Abnormal usage patterns
- **Compliance Monitoring**: Detect policy violations

## 🧪 Evaluation Methodology

### Synthetic Ground Truth
Since real network logs don't come with anomaly labels, we:
1. Generate synthetic anomalies with known characteristics
2. Compare model predictions against synthetic labels
3. Calculate approximate precision/recall metrics
4. Validate using domain expert knowledge

### Challenges with Unsupervised Evaluation
- **No True Labels**: Real anomalies are unknown
- **Subjective Definition**: What constitutes an anomaly varies
- **Temporal Dependencies**: Anomalies may be context-dependent
- **False Positive Trade-offs**: Balance between detection and noise

## 🛠️ API Endpoints

- `GET /status` - System status and configuration
- `POST /train` - Collect training data and train models
- `POST /start` - Start real-time inference
- `POST /stop` - Stop real-time inference
- `POST /model` - Switch active model
- `GET /metrics` - Current evaluation metrics
- `GET /models` - Available models and training status
- `WebSocket /ws` - Real-time data stream

## 🔧 Configuration

Key parameters can be adjusted in the code:

```python
# Feature Engineering
window_size_seconds = 30  # Time window for features

# Isolation Forest
contamination = 0.1  # Expected anomaly rate

# Autoencoder
threshold_percentile = 95  # Anomaly threshold
epochs = 50  # Training epochs

# Log Generation
anomaly_probability = 0.05  # 5% anomaly rate
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY backend/ /app/
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### Cloud Deployment
- **AWS**: ECS/Fargate with ALB
- **GCP**: Cloud Run with Cloud Load Balancing
- **Azure**: Container Instances with Application Gateway

## 🎓 Interview Talking Points

### Technical Depth
- **ML Pipeline Design**: End-to-end system thinking
- **Real-time Processing**: Streaming vs batch trade-offs
- **Model Selection**: Unsupervised learning challenges
- **Feature Engineering**: Domain knowledge application

### System Design
- **Scalability**: Horizontal scaling strategies
- **Reliability**: Error handling and recovery
- **Monitoring**: Observability and alerting
- **Performance**: Latency optimization techniques

### Business Impact
- **Cost Reduction**: Early problem detection
- **Service Quality**: Proactive monitoring
- **Operational Efficiency**: Automated anomaly detection
- **Risk Mitigation**: Security and compliance benefits

This system demonstrates production-ready ML engineering skills, combining theoretical knowledge with practical implementation in a real-world scenario.