import os
import time
import json
import uuid
import logging
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model_latest.pkl")
model_pipeline = None

# --- NEW: PROMETHEUS CUSTOM METRICS ---
PREDICTION_SCORE = Histogram('fraud_prediction_score', 'Distribution of fraud probabilities', buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
TRANSACTION_AMOUNT = Histogram('transaction_amount_usd', 'Distribution of transaction amounts', buckets=[10, 50, 100, 500, 1000, 5000])
FRAUD_FLAGGED_TOTAL = Counter('fraud_flagged_total', 'Total number of flagged transactions')

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    
    # Ensure data directory exists for structured logging
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"CRITICAL: Model file not found at {MODEL_PATH}. Halting startup.")
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    
    start_time = time.time()
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully in {time.time() - start_time:.3f} seconds.")
    
    yield
    logger.info("Shutting down API...")

# Initialize FastAPI App
app = FastAPI(
    title="Fraud Scoring API with Observability",
    description="Real-time API with Prometheus metrics and structured drift logging.",
    version="1.1.0",
    lifespan=lifespan
)

# Standard prometheus instrumentation for latency/HTTP errors
Instrumentator().instrument(app).expose(app)

class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: int = Field(..., ge=1, le=10, description="Merchant category code (1-10)")
    distance_from_home: float = Field(..., ge=0, description="Distance from home address in miles")
    high_risk_country: int = Field(..., ge=0, le=1, description="1 if high risk country, 0 otherwise")

@app.get("/health")
async def health_check():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_PATH}

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        features = [[
            transaction.amount,
            transaction.merchant_category,
            transaction.distance_from_home,
            transaction.high_risk_country
        ]]
        
        fraud_prob = float(model_pipeline.predict_proba(features)[0][1])
        is_fraud = bool(fraud_prob > 0.5)
        
        # --- NEW: UPDATE LIVE METRICS ---
        PREDICTION_SCORE.observe(fraud_prob)
        TRANSACTION_AMOUNT.observe(transaction.amount)
        if is_fraud:
            FRAUD_FLAGGED_TOTAL.inc()
        
        process_time = time.time() - start_time
        
        # --- NEW: STRUCTURED LOGGING FOR DRIFT CHECK ---
        log_entry = {
            "request_id": request_id,
            "timestamp": time.time(),
            "amount": transaction.amount,
            "merchant_category": transaction.merchant_category,
            "distance_from_home": transaction.distance_from_home,
            "high_risk_country": transaction.high_risk_country,
            "fraud_probability": round(fraud_prob, 4),
            "is_flagged": is_fraud,
            "latency_ms": round(process_time * 1000, 2)
        }
        
        # Append to a central JSON Lines file (Data Lake simulation)
        with open("data/prediction_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return {
            "request_id": request_id,
            "fraud_probability": round(fraud_prob, 4),
            "is_flagged": is_fraud,
            "latency_ms": round(process_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error")