import os
import time
import logging
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model_latest.pkl")
model_pipeline = None

# 1. FAIL FAST: Load model on startup, crash if missing.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"CRITICAL: Model file not found at {MODEL_PATH}. Halting startup.")
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    
    start_time = time.time()
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully in {time.time() - start_time:.3f} seconds.")
    
    yield
    # Cleanup on shutdown (if any)
    logger.info("Shutting down API...")

# Initialize FastAPI App
app = FastAPI(
    title="Fraud Scoring API",
    description="Real-time API for predicting transaction fraud.",
    version="1.0.0",
    lifespan=lifespan
)

# 2. PROMETHEUS METRICS: Expose /metrics for DevOps scraping
Instrumentator().instrument(app).expose(app)

# 3. DATA VALIDATION: Define expected input schema strictly
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: int = Field(..., ge=1, le=10, description="Merchant category code (1-10)")
    distance_from_home: float = Field(..., ge=0, description="Distance from home address in miles")
    high_risk_country: int = Field(..., ge=0, le=1, description="1 if high risk country, 0 otherwise")

@app.get("/health")
async def health_check():
    """Basic health check for load balancers."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_PATH}

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """Scores a single transaction for fraud probability in real-time."""
    start_time = time.time()
    
    try:
        # Convert Pydantic object to format expected by scikit-learn
        features = [[
            transaction.amount,
            transaction.merchant_category,
            transaction.distance_from_home,
            transaction.high_risk_country
        ]]
        
        # Predict probability of class 1 (Fraud)
        fraud_prob = model_pipeline.predict_proba(features)[0][1]
        is_fraud = bool(fraud_prob > 0.5)
        
        process_time = time.time() - start_time
        logger.info(f"Scored transaction. Prob: {fraud_prob:.3f}, Latency: {process_time:.4f}s")
        
        return {
            "fraud_probability": round(fraud_prob, 4),
            "is_flagged": is_fraud,
            "latency_ms": round(process_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error")