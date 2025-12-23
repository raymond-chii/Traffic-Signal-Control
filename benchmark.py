import os
import numpy as np
import json
import warnings
import ray
from ray.rllib.algorithms.ppo import PPO
from ppo import TrafficEnv

warnings.filterwarnings("ignore")

def run_evaluation(agent_type, checkpoint=None, episodes=5, action_override=None):
    print(f"\n--- Benchmarking {agent_type} ---")

    # Simulation config
    sim_config = {
        "net_file": os.path.join("simulation", "network.net.xml"),
        "route_file": os.path.join("simulation", "routes.rou.xml"),
        "add_file": os.path.join("simulation", "traffic_light.add.xml"),
        "max_steps": 1000,
        "gui": False  # Run without GUI for speed
    }

    env = TrafficEnv(sim_config)

    # Load PPO agent if needed
    if agent_type == "PPO":
        algo = PPO.from_checkpoint(os.path.abspath(checkpoint))

    stats = {
        "rewards": [],
        "collisions": [],
        "danger": [],
        "waiting": [],
        "protected_times": []
    }

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_metrics = {"r": 0, "c": 0, "d": 0, "w": 0}
        ep_protected_times = []

        while not done:
            # DECISION LOGIC
            if action_override is not None:
                # Fixed action (for fixed-time policies)
                action = action_override
            elif agent_type == "Random":
                # Random policy - helps validate collision detection
                action = np.random.randint(0, 9)
            elif agent_type == "PPO":
                action = algo.compute_single_action(obs, explore=False)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Track protected time for this action
            percentage = (action + 1) * 10  # 0→10%, 1→20%, ..., 8→90%
            protected_time = 45.0 * percentage / 100.0
            ep_protected_times.append(protected_time)

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            ep_metrics["r"] += reward
            ep_metrics["c"] += info["step_collisions"]
            ep_metrics["d"] += info["danger_events"]
            ep_metrics["w"] += info["waiting"]

        avg_protected = np.mean(ep_protected_times) if ep_protected_times else 22.5

        # Calculate average waiting per step (to match training metrics)
        num_steps = len(ep_protected_times) if ep_protected_times else 1
        avg_waiting = ep_metrics["w"] / max(num_steps, 1)

        print(f"  Episode {ep+1}: Collisions={ep_metrics['c']}, "
              f"Danger={ep_metrics['d']}, Waiting={ep_metrics['w']:.0f}, "
              f"AvgProtected={avg_protected:.1f}s")

        stats["rewards"].append(ep_metrics["r"])
        stats["collisions"].append(ep_metrics["c"])
        stats["danger"].append(ep_metrics["d"])
        stats["waiting"].append(avg_waiting)  # Use average waiting per step
        stats["protected_times"].append(avg_protected)

    env.close()

    # Calculate means
    results = {k: float(np.mean(v)) for k, v in stats.items()}

    # Add standard deviation for collisions to see variability
    results["collisions_std"] = float(np.std(stats["collisions"]))

    return results


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")

    from ray.tune.registry import register_env
    sim_config = {
        "net_file": os.path.join("simulation", "network.net.xml"),
        "route_file": os.path.join("simulation", "routes.rou.xml"),
        "add_file": os.path.join("simulation", "traffic_light.add.xml"),
        "max_steps": 1000,
        "gui": False
    }
    register_env("TrafficEnv", lambda cfg: TrafficEnv(cfg))

    print("\n" + "="*70)
    print("COMPREHENSIVE BENCHMARK: All Fixed Policies vs PPO")
    print("="*70)

    # First, test random policy to validate collision detection works
    print("\n VALIDATION: ")
    random_stats = run_evaluation("Random", episodes=3)

    print("\n" + "="*70)
    print("TESTING ALL FIXED-TIME POLICIES (10% - 90%)")
    print("="*70)

    # Test all 9 fixed-time policies
    fixed_policies = {}
    for action_id in range(9):
        percentage = (action_id + 1) * 10
        protected_time = 45.0 * percentage / 100.0
        permissive_time = 45.0 - protected_time

        policy_name = f"Fixed-{percentage}% ({protected_time:.1f}s/{permissive_time:.1f}s)"
        fixed_policies[policy_name] = run_evaluation(
            agent_type=f"Fixed-{percentage}%",
            episodes=5,
            action_override=action_id
        )

    # Test PPO
    print("\n" + "="*70)
    print("TESTING PPO (ADAPTIVE POLICY)")
    print("="*70)

    ppo_stats = None
    try:
        if os.path.exists("./ppo_best_checkpoint"):
            ppo_stats = run_evaluation("PPO", checkpoint="./ppo_best_checkpoint", episodes=5)
        else:
            print("\n  PPO checkpoint not found at ./ppo_best_checkpoint")
            print("   Run training first: python ppo.py --mode train")
    except Exception as e:
        print(f"\n  Could not load PPO checkpoint: {e}")

    # Compile results
    comparison = {
        "Random Policy (validation)": random_stats,
        **fixed_policies
    }
    if ppo_stats:
        comparison["PPO (Adaptive)"] = ppo_stats

    # Save to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)

    # Print in table format
    print(f"\n{'Policy':<30} | {'Reward':>8} | {'Collisions':>11} | {'Danger':>8} | {'Waiting':>10}")
    print("-" * 85)

    for agent, stats in comparison.items():
        collision_str = f"{stats['collisions']:.1f}"
        if stats.get('collisions_std', 0) > 0:
            collision_str += f" (±{stats['collisions_std']:.1f})"

        print(f"{agent:<30} | {stats['rewards']:8.2f} | {collision_str:>11} | "
              f"{stats['danger']:8.1f} | {stats['waiting']:10.2f}")

    # Analysis: Find best fixed policy
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Find best fixed policy by collisions
    fixed_only = {k: v for k, v in fixed_policies.items()}
    best_fixed_collisions = min(fixed_only.items(), key=lambda x: x[1]['collisions'])
    best_fixed_waiting = min(fixed_only.items(), key=lambda x: x[1]['waiting'])
    best_fixed_reward = max(fixed_only.items(), key=lambda x: x[1]['rewards'])

    print(f"\nBest Fixed Policy (Safety): {best_fixed_collisions[0]}")
    print(f"  Collisions: {best_fixed_collisions[1]['collisions']:.2f}")

    print(f"\nBest Fixed Policy (Efficiency): {best_fixed_waiting[0]}")
    print(f"  Waiting Time: {best_fixed_waiting[1]['waiting']:.2f}")

    print(f"\nBest Fixed Policy (Overall Reward): {best_fixed_reward[0]}")
    print(f"  Reward: {best_fixed_reward[1]['rewards']:.2f}")

    if ppo_stats:
        print(f"\nPPO Performance:")
        print(f"  Collisions: {ppo_stats['collisions']:.2f}")
        print(f"  Waiting Time: {ppo_stats['waiting']:.2f}")
        print(f"  Reward: {ppo_stats['rewards']:.2f}")

        # Compare PPO to best fixed policies
        print(f"\nPPO vs Best Fixed Policies:")

        collision_improvement = ((best_fixed_collisions[1]['collisions'] - ppo_stats['collisions']) /
                                 max(best_fixed_collisions[1]['collisions'], 0.001)) * 100
        print(f"  Collision improvement: {collision_improvement:+.1f}% vs best fixed")

        waiting_improvement = ((best_fixed_waiting[1]['waiting'] - ppo_stats['waiting']) /
                               best_fixed_waiting[1]['waiting']) * 100
        print(f"  Waiting time improvement: {waiting_improvement:+.1f}% vs best fixed")

        reward_improvement = ((ppo_stats['rewards'] - best_fixed_reward[1]['rewards']) /
                              abs(best_fixed_reward[1]['rewards'])) * 100
        print(f"  Reward improvement: {reward_improvement:+.1f}% vs best fixed")

    print("\n" + "="*70)
    print(" Full results saved to benchmark_results.json")
    print("="*70 + "\n")

    ray.shutdown()
