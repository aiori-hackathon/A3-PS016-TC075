# PS-016: Simulation-Based Handover Decision Evaluation for High-Mobility Users

This repository contains a reproducible 5G/LTE discrete-event handover evaluation testbed developed for AIORI-3. It implements and compares a conventional 3GPP Event A3 baseline with a supervised machine learning decision policy across high-mobility scenarios.

## 1. Architecture & Setup
- **Topology:** 3 linear cells along a highway with 800m inter-site distance (ISD).
- **Radio Model:** 3.5 GHz 5G NR carrier, log-distance path loss, and log-normal shadowing (std = 4 dB).
- **Speeds Tested:** 50 km/h (urban vehicular) and 130 km/h (high-speed expressway).
- **Evaluated Policies:**
  - **Baseline:** 3GPP Event A3 with Time-To-Trigger (TTT = 80ms) and Hysteresis (2.0 dB).
  - **ML Model:** Supervised Random Forest Classifier trained on signal trends, Doppler velocity, and dwell time.

## 2. Setup & Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Running the Pipeline
To train the classifier, run all multi-seed simulations, and generate metric plots:
```bash
python run_experiments.py
```

To run a fast verification demo:
```bash
python test_demo.py
```

## 4. Evaluated Metrics
All metrics are stored under `results/comparison_summary.csv` and plotted in `results/plots/`:
1. **Handover Failures (HOF)**
2. **Ping-Pong Handover Events**
3. **Connection Interruption Duration (ms)**
4. **Average Throughput (Mbps)**
