# Use slim Python image for smaller footprint
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/fraud_model_latest.pkl

# Set working directory
WORKDIR /app

# Install dependencies first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/

# Expose API port
EXPOSE 8000

# Create a non-root user for security best practices
RUN useradd -m apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Run the FastAPI application via Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]