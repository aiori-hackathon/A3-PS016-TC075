import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.simulator import HandoverSimulator
from src.baselines import EventA3Baseline
from src.ml_policy import SupervisedMLPolicy
from src.train import train_and_save
from src.mobility import TrajectoryGenerator

def run_rich_visual_demo():
    print("=" * 65)
    print("HANDOVER EVALUATION FOR HIGH-MOBILITY USERS")
    print("=" * 65)

    if not os.path.exists('models/random_forest_ho.pkl'):
        print("[!] Pre-trained model not found. Training model...")
        train_and_save()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    simulator = HandoverSimulator(config)
    baseline = EventA3Baseline(
        hysteresis_db=config['baseline_a3']['hysteresis_db'],
        time_to_trigger_s=config['baseline_a3']['time_to_trigger_s'],
        dt=config['simulation']['time_step_s']
    )
    ml_policy = SupervisedMLPolicy('models/random_forest_ho.pkl')

    test_speed = 130  # km/h
    test_seed = 101

    print(f"\n[*] Scenario Parameters:")
    print(f"    - Mobility Speed : {test_speed} km/h (Expressway Corridor)")
    print(f"    - Base Stations  : 3 Cells (Linear, ISD = 800m)")
    print(f"    - Radio Band     : Carrier = 3.5 GHz | Bandwidth = 20 MHz")
    print(f"    - Random Seed    : {test_seed}\n")

    # 1. Run simulation metrics
    baseline.reset()
    res_base = simulator.run_trajectory(baseline, speed_kmh=test_speed, seed=test_seed)
    res_ml = simulator.run_trajectory(ml_policy, speed_kmh=test_speed, seed=test_seed)

    # 2. Extract step trajectory for visualization
    dt = config['simulation']['time_step_s']
    def trace_handovers(policy_obj):
        rng_local = np.random.default_rng(test_seed)
        mob = TrajectoryGenerator(speed_kmh=test_speed, dt=dt, total_dist_m=config['simulation']['track_length_m'])
        serving = 0
        dwell = 0.0
        ho_points = []
        
        pos, _, done = mob.step()
        prev_r = {c: simulator.channel.compute_rsrp(pos, simulator.cell_coords[c], rng_local) for c in simulator.cell_coords}
        
        while not done:
            pos, _, done = mob.step()
            r = {c: simulator.channel.compute_rsrp(pos, simulator.cell_coords[c], rng_local) for c in simulator.cell_coords}
            
            if hasattr(policy_obj, 'feature_cols'):
                should_ho, target = policy_obj.decide(serving, r, prev_r, mob.speed_ms, dwell)
            else:
                should_ho, target = policy_obj.decide(serving, r)
                
            if should_ho and target != serving:
                ho_points.append((pos[0], r[target], serving, target))
                serving = target
                dwell = 0.0
            else:
                dwell += dt
            prev_r = r
        return ho_points

    base_ho_points = trace_handovers(baseline)
    ml_ho_points = trace_handovers(ml_policy)

    positions = []
    rsrp_c0, rsrp_c1, rsrp_c2 = [], [], []
    rng_plot = np.random.default_rng(test_seed)
    for x in np.linspace(0, 1600, 450):
        pos = np.array([x, 15.0])
        positions.append(x)
        rsrp_c0.append(simulator.channel.compute_rsrp(pos, simulator.cell_coords[0], rng_plot))
        rsrp_c1.append(simulator.channel.compute_rsrp(pos, simulator.cell_coords[1], rng_plot))
        rsrp_c2.append(simulator.channel.compute_rsrp(pos, simulator.cell_coords[2], rng_plot))

    # 3. Print terminal scorecard
    print("=" * 65)
    print(f"{'METRIC':<26} | {'3GPP A3 BASELINE':<16} | {'SUPERVISED ML':<16}")
    print("=" * 65)
    print(f"{'Total Handovers':<26} | {res_base['handovers']:<16} | {res_ml['handovers']:<16}")
    print(f"{'Ping-Pong Events':<26} | {res_base['ping_pongs']:<16} | {res_ml['ping_pongs']:<16}")
    print(f"{'Handover Failures':<26} | {res_base['failures']:<16} | {res_ml['failures']:<16}")
    print(f"{'Interruption Duration':<26} | {res_base['interruption_ms']:<13.1f} ms | {res_ml['interruption_ms']:<13.1f} ms")
    print(f"{'Average Throughput':<26} | {res_base['avg_throughput_mbps']:<11.2f} Mbps | {res_ml['avg_throughput_mbps']:<11.2f} Mbps")
    print("=" * 65)
    print("\n[+] Opening visual trajectory window...")

    # 4. Interactive Matplotlib plot
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title('PS-016: Live Handover Trajectory Visualizer')

    ax.plot(positions, rsrp_c0, label='Cell 0 (0m)', color='#1f77b4', alpha=0.45, linestyle=':')
    ax.plot(positions, rsrp_c1, label='Cell 1 (800m)', color='#ff7f0e', alpha=0.45, linestyle=':')
    ax.plot(positions, rsrp_c2, label='Cell 2 (1600m)', color='#2ca02c', alpha=0.45, linestyle=':')

    # Baseline markers with alternating vertical offset to prevent clashing
    for idx, (hx, hy, src, dst) in enumerate(base_ho_points):
        lbl = '3GPP A3 Trigger' if idx == 0 else ""
        ax.axvline(x=hx, color='red', linestyle='--', alpha=0.6)
        ax.scatter(hx, hy + 2, color='red', marker='v', s=80, label=lbl, zorder=5)
        y_offset = 4.0 if idx % 2 == 0 else 8.5
        ax.annotate(f"A3: C{src}->C{dst}", (hx, hy + y_offset), color='red', fontsize=8, ha='center', weight='bold')

    # ML markers
    for idx, (hx, hy, src, dst) in enumerate(ml_ho_points):
        lbl = 'ML Model Trigger' if idx == 0 else ""
        ax.axvline(x=hx, color='green', linestyle='-', alpha=0.8)
        ax.scatter(hx, hy - 2, color='green', marker='*', s=140, label=lbl, zorder=6)
        ax.annotate(f"ML: C{src}->C{dst}", (hx, hy - 6.0), color='green', fontsize=9, ha='center', weight='bold')

    # Base Station text tags (clean ASCII text, no missing glyphs)
    for bs_x, bs_id in [(0, 0), (800, 1), (1600, 2)]:
        ax.axvline(x=bs_x, color='gray', linestyle=':', alpha=0.5)
        ax.text(bs_x, -70, f"[BS {bs_id}]", horizontalalignment='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_title(f'RSRP Signal Profiles & Handover Decisions (Speed: {test_speed} km/h)', fontsize=12, weight='bold')
    ax.set_xlabel('Highway Position (meters)', fontsize=10)
    ax.set_ylabel('RSRP (dBm)', fontsize=10)
    ax.set_xlim(-50, 1650)
    ax.set_ylim(-125, -65)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(loc='lower center', ncol=3, frameon=True, fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_rich_visual_demo()