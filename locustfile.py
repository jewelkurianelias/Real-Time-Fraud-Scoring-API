from locust import HttpUser, task, between
import random

class FraudScoringUser(HttpUser):
    # Simulate a user hitting the API every 10 to 50 milliseconds
    wait_time = between(0.01, 0.05)

    @task(5)
    def predict_endpoint(self):
        """Simulate real-time transaction scoring."""
        payload = {
            "amount": round(random.uniform(5.0, 1500.0), 2),
            "merchant_category": random.randint(1, 10),
            "distance_from_home": round(random.uniform(0.0, 500.0), 2),
            "high_risk_country": random.choices([0, 1], weights=[0.9, 0.1])[0]
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Periodically hit the health endpoint like a load balancer would."""
        self.client.get("/health")