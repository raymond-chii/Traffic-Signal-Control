#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_training_results(checkpoint_dir="ppo_best_checkpoint"):
    results = []

    progress_file = os.path.join(checkpoint_dir, "progress.csv")

    if os.path.exists(progress_file):
        import pandas as pd
        df = pd.read_csv(progress_file)
        return df

    return None

def plot_training_curves(results_file="training_results.json"):

    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found!")
        print("Please run training with results logging or create the file manually.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    iterations = data['iterations']
    rewards = data['rewards']
    episode_lengths = data['episode_lengths']

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('PPO Training Progress - Traffic Signal Control', fontsize=16, fontweight='bold')

    # Plot 1: Reward over iterations
    ax1.plot(iterations, rewards, 'b-o', linewidth=2, markersize=6, label='Episode Reward')
    ax1.axhline(y=-440, color='r', linestyle='--', linewidth=2, label='Baseline (Always-Green)')
    ax1.set_xlabel('Training Iteration', fontsize=12)
    ax1.set_ylabel('Mean Episode Reward', fontsize=12)
    ax1.set_title('Learning Curve: Reward vs Iteration', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Add improvement annotation
    improvement = rewards[-1] - rewards[0]
    ax1.text(0.05, 0.95, f'Improvement: {improvement:+.1f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Episode length over iterations
    ax2.plot(iterations, episode_lengths, 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Iteration', fontsize=12)
    ax2.set_ylabel('Mean Episode Length (decisions)', fontsize=12)
    ax2.set_title('Episode Length Stability', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_progress.png")
    plt.show()

def plot_comparison(results_file="training_results.json"):
    """Create bar chart comparing final performance to baselines"""

    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found!")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    final_reward = data['rewards'][-1]

    # Baseline values (from your testing)
    methods = ['Always\nN-S', 'Always\nE-W', 'Random', 'Alternate\nEvery 5', 'PPO\n(Trained)']
    rewards = [-441, -436, -1252, -1179, final_reward]
    colors = ['lightcoral', 'lightcoral', 'lightgray', 'lightgray', 'lightgreen']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, rewards, color=colors, edgecolor='black', linewidth=1.5)

    # Highlight the best performer
    best_idx = np.argmax(rewards)
    bars[best_idx].set_edgecolor('darkgreen')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Mean Episode Reward', fontsize=13)
    ax.set_title('Performance Comparison: PPO vs Baseline Policies', fontsize=15, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_comparison.png")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Traffic Signal Control - Results Visualization")
    print("="*60 + "\n")

    # Check if results file exists
    if os.path.exists("training_results.json"):
        plot_training_curves()
        plot_comparison()
    else:
        print("No training_results.json found.")
        print("\nTo create plots, either:")
        print("  1. Wait for training to complete and create the results file")
        print("  2. Create training_results.json manually with format:")
        print('     {"iterations": [1,2,3...], "rewards": [-500,-450,...], "episode_lengths": [320,325,...]}')
