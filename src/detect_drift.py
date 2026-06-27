import pandas as pd
import json
import os

# Simulated Baseline metrics (What the model learned during training)
# If real-world traffic shifts significantly from these, the model becomes obsolete.
BASELINE_MEAN_AMOUNT = 50.0  
ALERT_THRESHOLD_PERCENT = 0.50  # We alert if the average transaction jumps by 50%

def send_alert(metric, old_val, new_val, change):
    """Simulates an automated webhook firing to a Slack/Teams channel."""
    print("\n" + "="*60)
    print("🚨 PAGERDUTY / SLACK ALERT TRIGGERED 🚨")
    print("="*60)
    print(f"MODEL DEGRADATION WARNING: Concept Drift Detected!")
    print(f"Metric Monitored:  {metric}")
    print(f"Expected Baseline: ${old_val:.2f}")
    print(f"Current Traffic:   ${new_val:.2f}")
    print(f"Calculated Shift:  +{change*100:.1f}%")
    print("\nAction Required: Check Grafana dashboards immediately.")
    print("Run `src/train.py` to retrain the model on fresh data distribution.")
    print("="*60 + "\n")

def run_drift_check():
    print("Running scheduled drift detection...")
    log_file = "data/prediction_logs.jsonl"
    
    if not os.path.exists(log_file):
        print("No prediction logs found. Run simulate_traffic.py first.")
        return

    # Ingest structured logs
    records = []
    with open(log_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        return

    df = pd.DataFrame(records)
    current_mean_amount = df["amount"].mean()

    # Calculate percentage drift
    amount_drift = abs(current_mean_amount - BASELINE_MEAN_AMOUNT) / BASELINE_MEAN_AMOUNT
    
    print(f"Current Avg Transaction: ${current_mean_amount:.2f}")
    
    # The Evaluation Gate
    if amount_drift > ALERT_THRESHOLD_PERCENT:
        send_alert("Average Transaction Amount", BASELINE_MEAN_AMOUNT, current_mean_amount, amount_drift)
    else:
        print("✅ Data distributions look healthy. No drift detected.")

if __name__ == "__main__":
    run_drift_check()