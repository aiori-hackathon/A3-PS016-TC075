import os
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.simulator import HandoverSimulator
from src.mobility import TrajectoryGenerator

def generate_training_data(config, n_trajectories=30, seed=42):
    rng = np.random.default_rng(seed)
    sim = HandoverSimulator(config)
    dt = config['simulation']['time_step_s']
    track_len = config['simulation']['track_length_m']
    
    speeds = [50.0, 90.0, 130.0]
    records = []
    
    for _ in range(n_trajectories):
        spd = float(rng.choice(speeds))
        mob = TrajectoryGenerator(speed_kmh=spd, dt=dt, total_dist_m=track_len)
        
        serving = 0
        dwell = 0.0
        pos, _, done = mob.step()
        prev_r = {c: sim.channel.compute_rsrp(pos, sim.cell_coords[c], rng) for c in sim.cell_coords}
        
        while not done:
            pos, _, done = mob.step()
            r = {c: sim.channel.compute_rsrp(pos, sim.cell_coords[c], rng) for c in sim.cell_coords}
            
            # Forward candidate cells (0 -> 1 -> 2)
            fwd = {c: p for c, p in r.items() if c > serving}
            if fwd:
                best_cand, best_rsrp = max(fwd.items(), key=lambda x: x[1])
                serv_rsrp = r[serving]
                delta_s = serv_rsrp - prev_r.get(serving, serv_rsrp)
                delta_t = best_rsrp - prev_r.get(best_cand, best_rsrp)
                diff = best_rsrp - serv_rsrp
                
                # Switch when candidate is stronger and user has dwelled at least 0.2s
                should_switch = 1 if (diff > 0.0 and dwell >= 0.2) else 0
                
                records.append({
                    'serving_rsrp': serv_rsrp,
                    'target_rsrp': best_rsrp,
                    'rsrp_diff': diff,
                    'delta_serving': delta_s,
                    'delta_target': delta_t,
                    'speed': mob.speed_ms,
                    'dwell_time': dwell,
                    'label': should_switch
                })
                
                if should_switch:
                    serving = best_cand
                    dwell = 0.0
                else:
                    dwell += dt
            else:
                dwell += dt
                
            prev_r = r

    df = pd.DataFrame(records)
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/train_features.csv', index=False)
    return df

def train_and_save():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    df = generate_training_data(config)
    
    features = ['serving_rsrp', 'target_rsrp', 'rsrp_diff', 
                'delta_serving', 'delta_target', 'speed', 'dwell_time']
    X = df[features]
    y = df['label']
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/random_forest_ho.pkl')
    print("[+] Model successfully trained and saved to models/random_forest_ho.pkl")

if __name__ == '__main__':
    train_and_save()
