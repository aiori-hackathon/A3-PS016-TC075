import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.simulator import HandoverSimulator
from src.baselines import EventA3Baseline
from src.ml_policy import SupervisedMLPolicy
from src.train import train_and_save

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if not os.path.exists('models/random_forest_ho.pkl'):
        print("[*] Pre-trained model not found. Training model...")
        train_and_save()

    simulator = HandoverSimulator(config)
    results = []

    print("\nRunning multi-seed comparative simulations...")

    for speed in config['simulation']['speeds_kmh']:
        for seed in config['simulation']['seeds']:
            # Re-initialize policy instances per seed to avoid state retention across runs
            ml_policy = SupervisedMLPolicy('models/random_forest_ho.pkl')
            baseline_policy = EventA3Baseline(
                hysteresis_db=config['baseline_a3']['hysteresis_db'],
                time_to_trigger_s=config['baseline_a3']['time_to_trigger_s'],
                dt=config['simulation']['time_step_s']
            )

            # 1. Evaluate 3GPP Event A3 Baseline
            base_metrics = simulator.run_trajectory(baseline_policy, speed_kmh=speed, seed=seed)
            base_metrics['policy'] = 'Baseline (3GPP A3)'
            base_metrics['speed_kmh'] = speed
            base_metrics['seed'] = seed
            results.append(base_metrics)

            # 2. Evaluate Supervised ML Policy
            ml_metrics = simulator.run_trajectory(ml_policy, speed_kmh=speed, seed=seed)
            ml_metrics['policy'] = 'Supervised ML'
            ml_metrics['speed_kmh'] = speed
            ml_metrics['seed'] = seed
            results.append(ml_metrics)

    df = pd.DataFrame(results)
    os.makedirs('results/plots', exist_ok=True)
    df.to_csv('results/raw_metrics.csv', index=False)

    # Compute Mean Metrics across Seeds
    summary = df.groupby(['speed_kmh', 'policy'])[[
        'failures', 'ping_pongs', 'interruption_ms', 'avg_throughput_mbps'
    ]].mean().reset_index()

    summary.to_csv('results/comparison_summary.csv', index=False)

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY (Mean across seeds)")
    print("=" * 65)
    print(summary.to_string(index=False))

    # Output Comparative Evaluation Charts
    sns.set_theme(style='whitegrid')

    # 1. Ping-Pong Events Plot
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=df, x='speed_kmh', y='ping_pongs', hue='policy', palette=['#d9534f', '#2ca02c'])
    plt.title('Ping-Pong Events vs Vehicular Speed (km/h)')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Ping-Pong Count')
    plt.tight_layout()
    plt.savefig('results/plots/ping_pong_vs_speed.png', dpi=300)
    plt.close()

    # 2. Average Throughput Plot
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=df, x='speed_kmh', y='avg_throughput_mbps', hue='policy', palette=['#d9534f', '#2ca02c'])
    plt.title('Average Throughput vs Vehicular Speed (km/h)')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Throughput (Mbps)')
    plt.tight_layout()
    plt.savefig('results/plots/throughput_comparison.png', dpi=300)
    plt.close()

    print("\n[+] Plots and tables successfully saved to results/ and results/plots/")

if __name__ == '__main__':
    main()