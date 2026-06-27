# Real-Time Fraud Scoring API

A containerized, production-ready REST API that serves a real-time machine learning fraud model.
This project focuses on MLOps best practices: fail-fast startup, strict input validation, basic telemetry, and containerization.

## 📁 Project Structure

```text
.
├── Dockerfile          # Container configuration
├── README.md           # This documentation
├── locustfile.py       # Load testing script (Locust)
├── requirements.txt    # Pinned Python dependencies
├── src/
│   ├── api.py          # FastAPI application and inference logic
│   └── train.py        # Data generation and model training script
├── models/             # Directory where trained models (.pkl) are saved
└── data/               # Directory where synthetic training data is stored
```

## 🚀 Quickstart

### 1. Train the Model

First, generate the synthetic data and train the artifact.

```bash
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/train.py
```

You should now see `models/fraud_model_latest.pkl`.

### 2. Run the API Locally

You can run it directly via Uvicorn to test it out:

```bash
uvicorn src.api:app --reload
```

Go to http://127.0.0.1:8000/docs to see the auto-generated Swagger UI and test the endpoint!

### 3. Build & Run via Docker (Production Simulation)

```bash
# Build the image
docker build -t fraud-api:latest .

# Run the container
docker run -p 8000:8000 fraud-api:latest
```

---

## 🧪 Testing "Fail Fast" Reliability

A major requirement of this project is avoiding silent failures.
Let's intentionally break the configuration to see our fail-fast logic in action.

Stop your Docker container, and run it again, but this time pass a broken file path using an environment variable:

```bash
docker run -p 8000:8000 -e MODEL_PATH=/app/models/does_not_exist.pkl fraud-api:latest
```

**Expected Outcome:** The container will immediately crash with a `FileNotFoundError`. This is good. In a production environment with Kubernetes, this prevents a pod from reporting as "healthy" when it actually has no model loaded to serve traffic.

---

## 📈 Load Testing & Latency

To simulate high traffic, we use Locust. Open a second terminal while your Docker container is running:

```bash
locust -f locustfile.py
```

Open http://0.0.0.0:8089 in your browser. Enter 100 users with a spawn rate of 10. Set the host to http://127.0.0.1:8000.

**Observed Latency Notes (2025 Hardware):**

* **Inference time:** The RandomForest inside FastAPI should execute in < 5ms.
* **Throughput:** A single Python worker can handle ~300-500 requests per second. For more throughput, you would scale horizontally (e.g., `docker run --cpus 4` and configuring Uvicorn with `--workers 4`).

---

## 📊 Telemetry

Because we included `prometheus-fastapi-instrumentator`, you can navigate to http://localhost:8000/metrics. This exposes standard operational metrics (request counts, latency histograms) that a DevOps team can instantly scrape into Grafana.
