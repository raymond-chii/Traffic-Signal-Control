import json
import os
import argparse
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6
})


def plot_training_metrics(json_file="training_results.json", output_dir="plots"):
    if not os.path.exists(json_file):
        print(f" Training results file not found: {json_file}")
        return

    with open(json_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    iterations = data["iterations"]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Progress: Traffic Signal Control", fontsize=16, fontweight='bold')

    # 1. Reward
    ax = axes[0, 0]
    ax.plot(iterations, data["rewards"], 'b-o', label='Episode Reward', alpha=0.8)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Reward")
    ax.set_title("Learning Progress (Reward)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Collisions
    ax = axes[0, 1]
    ax.plot(iterations, data["collisions"], 'r-s', label='Collisions', alpha=0.8)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Collisions per Episode")
    ax.set_title("Safety: Vehicle Collisions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0')

    # 3. Danger Events
    ax = axes[1, 0]
    ax.plot(iterations, data["danger"], 'orange', marker='^', label='Near-Miss Events', alpha=0.8)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Danger Events per Episode")
    ax.set_title("Safety: Vehicle-Pedestrian Near-Misses")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Waiting Time
    ax = axes[1, 1]
    ax.plot(iterations, data["avg_waiting"], 'g-d', label='Avg Waiting Time', alpha=0.8)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Waiting Time (seconds)")
    ax.set_title("Efficiency: Vehicle Waiting Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_progress.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training plot: {output_path}")
    plt.close()


def plot_benchmark_comparison(json_file="benchmark_results.json", output_dir="plots"):
    if not os.path.exists(json_file):
        print(f" Benchmark results file not found: {json_file}")
        print("   Run benchmark first: python benchmark.py")
        return

    with open(json_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    agents = list(data.keys())
    metrics = ["collisions", "danger", "waiting", "rewards"]
    metric_labels = ["Collisions\n(lower is better)",
                     "Near-Miss Events\n(lower is better)",
                     "Waiting Time (s)\n(lower is better)",
                     "Total Reward\n(higher is better)"]

    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Comparison: PPO vs Fixed-Time Control",
                 fontsize=16, fontweight='bold')

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red, Orange, Green, Blue

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        values = [data[agent][metric] for agent in agents]

        # Highlight PPO bar
        bar_colors = ['lightgray'] * (len(agents) - 1) + [colors[idx]]
        bars = ax.bar(range(len(agents)), values, color=bar_colors, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([a.replace(' ', '\n') for a in agents], fontsize=9)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved benchmark plot: {output_path}")
    plt.close()

    # Create improvement percentage table
    if "PPO (Adaptive)" in data:
        print("\n" + "="*70)
        print("PPO IMPROVEMENT OVER FIXED-TIME BASELINES")
        print("="*70)

        fixed_baselines = [k for k in agents if k.startswith("Fixed-Time")]
        ppo_metrics = data["PPO (Adaptive)"]

        for baseline in fixed_baselines:
            baseline_metrics = data[baseline]
            print(f"\n{baseline}:")

            for metric in ["collisions", "danger", "waiting"]:
                baseline_val = baseline_metrics[metric]
                ppo_val = ppo_metrics[metric]
                if baseline_val > 0:
                    improvement = ((baseline_val - ppo_val) / baseline_val) * 100
                    print(f"  {metric.capitalize():20s}: {improvement:+6.1f}% "
                          f"({baseline_val:.1f} → {ppo_val:.1f})")

            # Reward is opposite (higher is better)
            baseline_reward = baseline_metrics["rewards"]
            ppo_reward = ppo_metrics["rewards"]
            if abs(baseline_reward) > 0:
                improvement = ((ppo_reward - baseline_reward) / abs(baseline_reward)) * 100
                print(f"  {'Reward':20s}: {improvement:+6.1f}% "
                      f"({baseline_reward:.1f} → {ppo_reward:.1f})")


def create_summary_plot(output_dir="plots"):
    training_file = "training_results.json"
    benchmark_file = "benchmark_results.json"

    if not (os.path.exists(training_file) and os.path.exists(benchmark_file)):
        print(" Need both training_results.json and benchmark_results.json")
        return

    with open(training_file) as f:
        training_data = json.load(f)
    with open(benchmark_file) as f:
        benchmark_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle("PPO Traffic Signal Control: Complete Analysis",
                 fontsize=18, fontweight='bold')

    # Training metrics (top row)
    iterations = training_data["iterations"]

    # Reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, training_data["rewards"], 'b-o', alpha=0.8)
    ax1.set_title("Training: Reward Progress")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    # Collisions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, training_data["collisions"], 'r-s', alpha=0.8)
    ax2.set_title("Training: Collision Reduction")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Collisions")
    ax2.grid(True, alpha=0.3)

    # Waiting Time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iterations, training_data["avg_waiting"], 'g-d', alpha=0.8)
    ax3.set_title("Training: Waiting Time Optimization")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Waiting Time (s)")
    ax3.grid(True, alpha=0.3)

    # Benchmark comparisons (bottom row)
    agents = list(benchmark_data.keys())

    # Collisions comparison
    ax4 = fig.add_subplot(gs[1, 0])
    collisions = [benchmark_data[a]["collisions"] for a in agents]
    colors = ['lightgray'] * (len(agents) - 1) + ['red']
    ax4.bar(range(len(agents)), collisions, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(agents)))
    ax4.set_xticklabels([a.split()[0] for a in agents], rotation=45, ha='right', fontsize=8)
    ax4.set_title("Benchmark: Collisions")
    ax4.set_ylabel("Count")
    ax4.grid(axis='y', alpha=0.3)

    # Danger comparison
    ax5 = fig.add_subplot(gs[1, 1])
    danger = [benchmark_data[a]["danger"] for a in agents]
    colors = ['lightgray'] * (len(agents) - 1) + ['orange']
    ax5.bar(range(len(agents)), danger, color=colors, edgecolor='black')
    ax5.set_xticks(range(len(agents)))
    ax5.set_xticklabels([a.split()[0] for a in agents], rotation=45, ha='right', fontsize=8)
    ax5.set_title("Benchmark: Near-Misses")
    ax5.set_ylabel("Count")
    ax5.grid(axis='y', alpha=0.3)

    # Waiting comparison
    ax6 = fig.add_subplot(gs[1, 2])
    waiting = [benchmark_data[a]["waiting"] for a in agents]
    colors = ['lightgray'] * (len(agents) - 1) + ['green']
    ax6.bar(range(len(agents)), waiting, color=colors, edgecolor='black')
    ax6.set_xticks(range(len(agents)))
    ax6.set_xticklabels([a.split()[0] for a in agents], rotation=45, ha='right', fontsize=8)
    ax6.set_title("Benchmark: Waiting Time")
    ax6.set_ylabel("Seconds")
    ax6.grid(axis='y', alpha=0.3)

    output_path = os.path.join(output_dir, "complete_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved complete analysis: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO traffic control results")
    parser.add_argument("--training", action="store_true", help="Plot training metrics")
    parser.add_argument("--benchmark", action="store_true", help="Plot benchmark comparison")
    parser.add_argument("--summary", action="store_true", help="Create summary plot")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")

    args = parser.parse_args()

    # If no specific flag, generate all
    if not (args.training or args.benchmark or args.summary or args.all):
        args.all = True

    print("\n" + "="*70)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("="*70 + "\n")

    if args.all or args.training:
        plot_training_metrics(output_dir=args.output_dir)

    if args.all or args.benchmark:
        plot_benchmark_comparison(output_dir=args.output_dir)

    if args.all or args.summary:
        create_summary_plot(output_dir=args.output_dir)

    print("\n" + "="*70)
    print(f"✓ All plots saved to '{args.output_dir}/' directory")
    print("="*70 + "\n")
