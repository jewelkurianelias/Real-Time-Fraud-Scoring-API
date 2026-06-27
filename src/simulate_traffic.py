import requests
import random
import time

API_URL = "http://localhost:8000/predict"

def send_traffic(scenario="normal", num_requests=50):
    print(f"Simulating {num_requests} {scenario} requests...")
    
    for i in range(num_requests):
        if scenario == "normal":
            # Normal bounds (matches training data)
            amount = random.uniform(5.0, 150.0)
            distance = random.uniform(0.0, 50.0)
        elif scenario == "drift":
            # Sudden inflation/market shift: Massive transactions
            amount = random.uniform(1000.0, 5000.0)
            distance = random.uniform(200.0, 1000.0)
            
        payload = {
            "amount": round(amount, 2),
            "merchant_category": random.randint(1, 10),
            "distance_from_home": round(distance, 2),
            "high_risk_country": random.choices([0, 1], weights=[0.9, 0.1])[0]
        }
        
        try:
            res = requests.post(API_URL, json=payload)
            print(f"[{i+1}/{num_requests}] Status: {res.status_code} | Flagged: {res.json().get('is_flagged')}")
            time.sleep(0.1) # Sleep to create a realistic time-series curve
        except Exception as e:
            print("Failed to connect to API. Is it running?")
            break

if __name__ == "__main__":
    print("1. Sending Normal Traffic...")
    send_traffic("normal", 30)
    
    print("\n2. Sending Drifted Traffic (Market Shift)...")
    send_traffic("drift", 30)
    
    print("\n✅ Traffic simulation complete! Run `python src/detect_drift.py` to check for alerts.")