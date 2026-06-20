import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def generate_mock_data(n_samples=5000):
    """Generates synthetic credit card transactions."""
    np.random.seed(42)
    data = {
        "amount": np.random.exponential(scale=50, size=n_samples),
        "merchant_category": np.random.randint(1, 10, size=n_samples),
        "distance_from_home": np.random.exponential(scale=20, size=n_samples),
        "high_risk_country": np.random.binomial(1, 0.05, size=n_samples),
    }
    df = pd.DataFrame(data)
    
    # Simple logic to generate labels: 
    # High amount + high risk country + far from home = likely fraud
    fraud_prob = (
        (df["amount"] > 200).astype(int) * 0.4 + 
        df["high_risk_country"] * 0.4 + 
        (df["distance_from_home"] > 100).astype(int) * 0.2
    )
    df["is_fraud"] = (np.random.rand(n_samples) < fraud_prob).astype(int)
    return df

def train():
    print("Generating training data...")
    df = generate_mock_data()
    df.to_csv("data/training_data.csv", index=False)
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    print("Training Random Forest model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    # Save model with versioned name and update the 'latest' symlink/copy
    timestamp = int(time.time())
    versioned_path = f"models/fraud_model_v{timestamp}.pkl"
    latest_path = "models/fraud_model_latest.pkl"
    
    joblib.dump(pipeline, versioned_path)
    joblib.dump(pipeline, latest_path) # Overwrite latest
    
    print(f"✅ Model trained successfully.")
    print(f"✅ Versioned model saved to: {versioned_path}")
    print(f"✅ Latest model saved to: {latest_path}")

if __name__ == "__main__":
    train()