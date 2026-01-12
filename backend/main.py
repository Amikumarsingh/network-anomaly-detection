from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import json
from typing import List, Dict, Any
import uvicorn

from inference import InferenceEngine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Start background tasks
    broadcaster_task = asyncio.create_task(websocket_broadcaster())
    yield
    # Cleanup
    broadcaster_task.cancel()

app = FastAPI(
    title="Network Anomaly Detection System", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = InferenceEngine()
connected_websockets: List[WebSocket] = []

class ModelRequest(BaseModel):
    model_name: str

class TrainingRequest(BaseModel):
    duration_seconds: int = 120

from datetime import datetime
import json
import numpy as np

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# WebSocket connection manager
async def broadcast_to_websockets(data: Dict[str, Any]):
    """Broadcast data to all connected WebSocket clients"""
    if connected_websockets:
        message = json.dumps(data, cls=DateTimeEncoder)
        disconnected = []
        
        for websocket in connected_websockets:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            connected_websockets.remove(ws)

# WebSocket broadcasting
websocket_queue = asyncio.Queue()

async def websocket_broadcaster():
    """Background task to handle WebSocket broadcasting"""
    while True:
        try:
            data = await websocket_queue.get()
            await broadcast_to_websockets(data)
        except Exception as e:
            print(f"Broadcast error: {e}")

# Add callback to inference engine
def queue_broadcast(data):
    """Queue data for WebSocket broadcast"""
    try:
        websocket_queue.put_nowait(data)
    except Exception as e:
        print(f"Queue error: {e}")

inference_engine.add_callback(queue_broadcast)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

@app.get("/")
async def root():
    return {"message": "Network Anomaly Detection System API"}

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "is_running": inference_engine.is_running,
        "current_model": inference_engine.current_model,
        "training_data_size": len(inference_engine.training_data),
        "connected_clients": len(connected_websockets)
    }

@app.post("/train")
async def train_models(request: TrainingRequest):
    """Collect training data and train models"""
    try:
        # Stop inference if running
        if inference_engine.is_running:
            inference_engine.stop_inference()
            await asyncio.sleep(1)
        
        # Collect training data
        await inference_engine.collect_training_data(request.duration_seconds)
        
        # Train models
        results = inference_engine.train_models()
        
        return {
            "status": "success",
            "training_results": results,
            "training_samples": len(inference_engine.training_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
async def start_inference():
    """Start real-time inference"""
    if inference_engine.is_running:
        return {"message": "Inference already running"}
    
    if len(inference_engine.training_data) < 50:
        raise HTTPException(status_code=400, detail="Models not trained. Run /train first.")
    
    # Start inference in background
    asyncio.create_task(inference_engine.start_inference())
    
    return {"message": "Inference started"}

@app.post("/stop")
async def stop_inference():
    """Stop real-time inference"""
    inference_engine.stop_inference()
    return {"message": "Inference stopped"}

@app.post("/model")
async def set_model(request: ModelRequest):
    """Switch active model"""
    try:
        inference_engine.set_model(request.model_name)
        return {"message": f"Switched to {request.model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get evaluation metrics"""
    return inference_engine.get_metrics()

@app.get("/recent-results")
async def get_recent_results():
    """Get recent inference results"""
    recent = inference_engine.inference_results[-50:]  # Last 50 results
    return {"results": recent}

@app.get("/models")
async def get_available_models():
    """Get available models and their status"""
    return {
        "models": {
            "isolation_forest": {
                "name": "Isolation Forest",
                "trained": inference_engine.isolation_forest.is_trained,
                "description": "Unsupervised outlier detection using isolation trees"
            },
            "autoencoder": {
                "name": "Autoencoder",
                "trained": inference_engine.autoencoder.is_trained,
                "description": "Neural network that detects anomalies via reconstruction error"
            }
        },
        "current_model": inference_engine.current_model
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)