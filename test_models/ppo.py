#!/usr/bin/env python3
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci

# Ray and RLlib imports
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
import ray

# Import the Grid class
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# Import grid from the local module
from grid import Grid


class TrafficEnv(gym.Env):
    """
    Custom SUMO traffic environment for PPO training.
    Uses grid-based state representation from grid.py
    """
    
    def __init__(self, config: EnvContext):
        super().__init__()
        
        # Configuration
        self.gui = config.get("gui", False)
        self.max_steps = config.get("max_steps", 600)
        self.action_duration = config.get("action_duration", 15)
        self.grid_size = config.get("grid_size", (8, 8))
        self.bounds = config.get("bounds", (120, 120, 280, 280))
        
        # SUMO configuration
        self.net_file = config.get("net_file", "simulation/network.net.xml")
        self.route_file = config.get("route_file", "simulation/routes.rou.xml")
        
        # Initialize grid observer
        self.grid = Grid(grid_size=self.grid_size, bounds=self.bounds)
        
        # Action space: 2 actions (North-South green or East-West green)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: flattened grid observations
        # 4 channels * 8 * 8 = 256 features
        rows, cols = self.grid_size
        self.observation_space = spaces.Box(
            low=0,
            high=100,  # Reasonable upper bound for grid cell counts/speeds
            shape=(4 * rows * cols,),  # Flatten to 1D vector
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.sumo_running = False
        self.total_waiting_time = 0.0
        self.last_action_step = 0
        
    def _start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--start", "true"
        ]
        import time
        traci.start(sumo_cmd)
        time.sleep(0.1)  # Give SUMO a moment
        self.sumo_running = True
        
    def _get_waiting_time(self):
        """Get total waiting time from all incoming edges"""
        edges = [
            "north_near_to_center",
            "south_near_to_center", 
            "east_near_to_center",
            "west_near_to_center"
        ]
        
        total_wait = 0.0
        for edge in edges:
            try:
                total_wait += traci.edge.getWaitingTime(edge)
            except traci.exceptions.TraCIException:
                pass
        
        return total_wait
    
    def _get_queue_length(self):
        """Get total number of halting vehicles"""
        edges = [
            "north_near_to_center",
            "south_near_to_center",
            "east_near_to_center", 
            "west_near_to_center"
        ]
        
        total_halting = 0
        for edge in edges:
            try:
                total_halting += traci.edge.getLastStepHaltingNumber(edge)
            except traci.exceptions.TraCIException:
                pass
                
        return total_halting
    
    def _apply_action(self, action):
        """Apply traffic light action"""
        # Action 0: North-South green (phase 0)
        # Action 1: East-West green (phase 5)
        if action == 0:
            traci.trafficlight.setPhase("center", 0)
        else:
            traci.trafficlight.setPhase("center", 5)
    
    def _get_observation(self):
        """Get grid-based observation, flattened to 1D"""
        obs = self.grid.get_stacked_observation()
        
        # Normalize speeds (assuming max speed is ~14 m/s from network)
        obs[2] = obs[2] / 14.0
        
        # Clip values to reasonable range
        obs = np.clip(obs, 0, 100)
        
        # Flatten from (4, rows, cols) to (4*rows*cols,)
        return obs.flatten().astype(np.float32)
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Close existing SUMO instance if running
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
        
        # Start new SUMO instance
        try:
            self._start_sumo()
            print(f"[TrafficEnv] SUMO started successfully with files:")
            print(f"  Network: {self.net_file}")
            print(f"  Routes: {self.route_file}")
        except Exception as e:
            print(f"[TrafficEnv ERROR] Failed to start SUMO: {e}")
            print(f"  Network file: {self.net_file}")
            print(f"  Routes file: {self.route_file}")
            raise
        
        # Reset tracking variables
        self.current_step = 0
        self.total_waiting_time = 0.0
        self.last_action_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action at the start of action duration
        if self.current_step % self.action_duration == 0:
            self._apply_action(action)
            self.last_action_step = self.current_step
        
        # Run simulation steps for the action duration
        step_waiting_time = 0.0
        for _ in range(self.action_duration):
            traci.simulationStep()
            self.current_step += 1
            
            # Accumulate waiting time
            step_waiting_time += self._get_waiting_time()
            
            # Check if simulation ended or max steps reached
            if traci.simulation.getMinExpectedNumber() <= 0 or self.current_step >= self.max_steps:
                break
        
        self.total_waiting_time += step_waiting_time
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward (negative waiting time)
        reward = -step_waiting_time
        
        # Check if episode is done
        terminated = (traci.simulation.getMinExpectedNumber() <= 0) or (self.current_step >= self.max_steps)
        truncated = False
        
        # Additional info
        info = {
            "total_waiting_time": self.total_waiting_time,
            "queue_length": self._get_queue_length(),
            "step": self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up environment"""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False


def train_ppo_agent(
    num_iterations=100,
    checkpoint_freq=10,
    num_workers=1,
    use_gpu=False
):
    """
    Train a PPO agent for traffic signal control.
    
    Args:
        num_iterations: Number of training iterations
        checkpoint_freq: Frequency of checkpoints
        num_workers: Number of parallel workers
        use_gpu: Whether to use GPU
    """
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register the custom environment
    from ray.tune.registry import register_env
    
    def env_creator(env_config):
        return TrafficEnv(env_config)
    
    register_env("TrafficEnv", env_creator)
    
    # Environment configuration
    env_config = {
        "gui": False,
        "max_steps": 3600,
        "action_duration": 15,
        "grid_size": (8, 8),
        "bounds": (120, 120, 280, 280),
        "net_file": "simulation/network.net.xml",
        "route_file": "simulation/routes.rou.xml"
    }
    
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            env="TrafficEnv",
            env_config=env_config
        )
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1
        )
        .training(
            train_batch_size=600,
            minibatch_size=128,
            num_epochs=10,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            model={
                # Simple fully connected network for flattened grid input (256 features)
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            }
        )
        # .resources(
        #     num_gpus=1 if use_gpu else 0,
        #     num_cpus_per_env_runner=1
        # )
        # .debugging(
        #     log_level="WARN"
        # )
    )
    
    # Build algorithm
    algo = config.build_algo()
    
    print("\n" + "="*60)
    print("Starting PPO Training")
    print("="*60)
    print(f"Iterations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"GPU: {use_gpu}")
    print("="*60 + "\n")
    
    best_reward = float('-inf')
    
    # Training loop
    for iteration in range(num_iterations):
        result = algo.train()
        
        # Extract key metrics - CORRECTED
        episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
        episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)
        
        print(f"Iteration {iteration + 1:3d}/{num_iterations} | "
            f"Reward: {episode_reward_mean:10.2f} | "
            f"Episode Length: {episode_len_mean:6.1f}")
        
        # Save checkpoint if best so far
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_path = algo.save("./ppo_traffic_checkpoints")
            print(f"  → New best! Checkpoint saved to: {checkpoint_path}")
        
        # Periodic checkpoint
        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save("./ppo_traffic_checkpoints")
            print(f"  → Checkpoint saved to: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best reward: {best_reward:.2f}")
    
    # Save final checkpoint
    final_checkpoint = algo.save("./ppo_traffic_checkpoints")
    print(f"Final checkpoint: {final_checkpoint}")
    
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


def test_ppo_agent(checkpoint_path, num_episodes=5):
    """
    Test a trained PPO agent.
    
    Args:
        checkpoint_path: Path to the checkpoint
        num_episodes: Number of test episodes
    """
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register the custom environment
    from ray.tune.registry import register_env
    
    def env_creator(env_config):
        return TrafficEnv(env_config)
    
    register_env("TrafficEnv", env_creator)
    
    # Environment configuration
    env_config = {
        "gui": False,  # Show GUI for testing
        "max_steps": 600,
        "action_duration": 15,
        "grid_size": (8, 8),
        "bounds": (120, 120, 280, 280),
        "net_file": "simulation/network.net.xml",
        "route_file": "simulation/routes.rou.xml"
    }
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Load algorithm
    from ray.rllib.algorithms.ppo import PPO
    algo = PPO.from_checkpoint(checkpoint_path)
    
    print("\n" + "="*60)
    print("Testing PPO Agent")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*60 + "\n")
    
    # Create test environment
    env = TrafficEnv(env_config)
    
    total_rewards = []
    total_waiting_times = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        while not done:
            # Get action from trained policy
            action = algo.compute_single_action(obs, explore=False)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Print periodic updates
            if info["step"] % 100 == 0:
                print(f"  Step {info['step']:3d} | "
                      f"Action: {action} | "
                      f"Queue: {info['queue_length']:3d} | "
                      f"Wait: {info['total_waiting_time']:8.2f}")
        
        total_rewards.append(episode_reward)
        total_waiting_times.append(info["total_waiting_time"])
        
        print(f"  Episode Reward: {episode_reward:.2f}")
        print(f"  Total Waiting Time: {info['total_waiting_time']:.2f} seconds")
    
    env.close()
    algo.stop()
    ray.shutdown()
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Waiting Time: {np.mean(total_waiting_times):.2f} ± {np.std(total_waiting_times):.2f}")
    print("="*60)
    
    return total_rewards, total_waiting_times


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Traffic Signal Control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for testing")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        checkpoint = train_ppo_agent(
            num_iterations=args.iterations,
            num_workers=args.workers,
            use_gpu=args.gpu
        )
        print(f"\nTo test this model, run:")
        print(f"python PPO_agent.py --mode test --checkpoint {checkpoint}")
        
    elif args.mode == "test":
        if args.checkpoint is None:
            print("Error: --checkpoint required for testing")
            sys.exit(1)
        
        test_ppo_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes
        )