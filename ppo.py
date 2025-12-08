#!/usr/bin/env python3
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from grid import Grid
except ImportError:
    print(f"WARNING: Could not import Grid from {SCRIPT_DIR}. Checking current directory.")
    try:
        from grid import Grid
    except ImportError:
        raise ImportError("Could not locate grid.py. Ensure it is in the same directory as ppo.py")

class TrafficEnv(gym.Env):
    
    def __init__(self, config: EnvContext):
        super().__init__()
        
        self.gui = config.get("gui", False)
        self.max_steps = config.get("max_steps", 7500)
        self.action_duration = config.get("action_duration", 15)
        self.yellow_duration = 3
        self.protected_duration = 10 
        
        self.grid_size = config.get("grid_size", (8, 8))
        self.bounds = config.get("bounds", (120, 120, 280, 280))
        

        self.net_file = config.get("net_file")
        self.route_file = config.get("route_file")
        self.add_file = config.get("add_file")
        
        if not os.path.exists(self.net_file):
            raise FileNotFoundError(f"Worker could not find network file at: {self.net_file}")

        self.grid = Grid(grid_size=self.grid_size, bounds=self.bounds)
        self.action_space = spaces.Discrete(2)
        
        rows, cols = self.grid_size
        input_dim = 4 * rows * cols
        
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(input_dim,), dtype=np.float32
        )
        
        self.sumo_running = False
        self.current_step = 0
        self.total_collisions = 0
        self.last_action = 1 
        
    def _start_sumo(self):
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.add_file,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--start", "true"
        ]
        traci.start(sumo_cmd)
        self.sumo_running = True
        
        try:
            # Force phase logic
            traci.trafficlight.setProgram("center", "protective_permitted_paper")
            traci.trafficlight.setPhase("center", 2)
        except:
            pass

    def _get_observation(self):
        obs = self.grid.get_stacked_observation()
        obs[2] = obs[2] / 14.0
        return obs.flatten().astype(np.float32)

    def _get_metrics(self):
        collisions = 0
        waiting = 0
        ped_waiting = 0
        
        try:
            # 1. Collision Check (Distance based)
            veh_ids = traci.vehicle.getIDList()
            ped_ids = traci.person.getIDList()
            
            if len(ped_ids) > 0 and len(veh_ids) > 0:
                for ped in ped_ids:
                    try:
                        p_pos = np.array(traci.person.getPosition(ped))
                        for veh in veh_ids:
                            v_pos = np.array(traci.vehicle.getPosition(veh))
                            if np.linalg.norm(p_pos - v_pos) < 2.0:
                                if traci.vehicle.getSpeed(veh) > 0.1:
                                    collisions += 1
                    except:
                        pass

            # 2. Waiting Time
            for edge in traci.edge.getIDList():
                waiting += traci.edge.getWaitingTime(edge)
                
            # 3. Pedestrian Waiting
            for ped in ped_ids:
                try:
                    if traci.person.getSpeed(ped) < 0.1:
                        ped_waiting += 1
                except:
                    pass
        except:
            pass
            
        return collisions, waiting, ped_waiting

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
        self._start_sumo()
        self.current_step = 0
        self.total_collisions = 0
        self.last_action = 1 
        return self._get_observation(), {}

    def step(self, action):
        step_collisions = 0
        step_waiting = 0
        step_ped_waiting = 0
        
        # --- PHASE SWITCHING ---
        if action != self.last_action:
            transitions = []
            target_phase = 0
            
            if self.last_action == 1: 
                transitions = [(3, self.yellow_duration), (4, self.protected_duration), (5, self.yellow_duration)]
                target_phase = 6
            else: 
                transitions = [(7, self.yellow_duration), (0, self.protected_duration), (1, self.yellow_duration)]
                target_phase = 2

            for phase_idx, duration in transitions:
                traci.trafficlight.setPhase("center", phase_idx)
                for _ in range(duration):
                    traci.simulationStep()
                    self.current_step += 1
                    c, w, pw = self._get_metrics()
                    step_collisions += c
                    step_waiting += w
                    step_ped_waiting += pw
            
            traci.trafficlight.setPhase("center", target_phase)
            self.last_action = action

        # --- EXECUTE ACTION ---
        for _ in range(self.action_duration):
            traci.simulationStep()
            self.current_step += 1
            c, w, pw = self._get_metrics()
            step_collisions += c
            step_waiting += w
            step_ped_waiting += pw
            if self.current_step >= self.max_steps:
                break
        
        # Penalties
        w_collision = 50.0
        w_car_wait = 0.01
        w_ped_wait = 0.05
        
        raw_penalty = (step_collisions * w_collision) + \
                      (step_waiting * w_car_wait) + \
                      (step_ped_waiting * w_ped_wait)
        
        reward = -raw_penalty * 0.1
        self.total_collisions += step_collisions
        
        obs = self._get_observation()
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "collisions": self.total_collisions,
            "waiting": step_waiting
        }
        
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

# --- HELPER TO RESOLVE PATHS ONCE ---
def get_sim_config():
    # Resolve paths relative to THIS script (which we know is correct)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(base_dir, "simulation")
    
    # Define absolute paths
    config = {
        "net_file": os.path.join(sim_dir, "network.net.xml"),
        "route_file": os.path.join(sim_dir, "routes.rou.xml"),
        "add_file": os.path.join(sim_dir, "traffic_light.add.xml"),
        "gui": False,
        "action_duration": 15
    }
    
    # Validation before we even send to workers
    if not os.path.exists(config["net_file"]):
        print(f"ERROR: Network file not found at: {config['net_file']}")
        print(f"Current Directory: {os.getcwd()}")
        print(f"Script Directory: {base_dir}")
        sys.exit(1)
        
    return config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    ray.init(
        ignore_reinit_error=True,
        logging_level="ERROR",
        object_store_memory=1e9,  # Limit object store to 1GB
        _temp_dir="/tmp/ray_traffic"  # Use custom temp dir
    )
    register_env("TrafficEnv", lambda config: TrafficEnv(config))

    # Calculate paths here in main process
    env_config_dict = get_sim_config()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            env="TrafficEnv",
            env_config=env_config_dict # Pass specific paths to workers
        )
        .framework("torch")
        .training(
            train_batch_size=1000,  # Reduced for CPU
            lr=5e-5,
            gamma=0.99,
            num_sgd_iter=10,  # Fewer SGD iterations for speed
            model={
                "fcnet_hiddens": [256, 256, 128],  # Keep original network size for performance
                "fcnet_activation": "relu"
            }
        )
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=50,  # Restore original rollout length
            sample_timeout_s=300.0
        )
        # CPU only - no GPU
    )

    if args.mode == "train":
        algo = config.build()
        print("\n" + "=" * 60)
        print("Starting PPO Training - CPU Optimized")
        print("=" * 60 + "\n")

        best_reward = float('-inf')
        training_results = {"iterations": [], "rewards": [], "episode_lengths": []}

        for i in range(30):  # Reduced from 50 for faster CPU training
            result = algo.train()

            # Handle different result key formats in Ray 2.9.0
            reward = result.get('env_runners', {}).get('episode_reward_mean') or \
                     result.get('episode_reward_mean', 0)
            ep_len = result.get('env_runners', {}).get('episode_len_mean') or \
                     result.get('episode_len_mean', 0)

            training_results["iterations"].append(i + 1)
            training_results["rewards"].append(float(reward))
            training_results["episode_lengths"].append(float(ep_len))

            print(f"Iter {i+1:2d}/30 | Reward: {reward:8.2f} | Ep Length: {ep_len:6.1f}")

            if reward > best_reward:
                best_reward = reward
                checkpoint = algo.save("./ppo_best_checkpoint")
                print(f"         → New Best! Saved.")

        import json
        with open("training_results.json", "w") as f:
            json.dump(training_results, f, indent=2)
        print(f"\n✓ Training complete! Results saved.")
        print(f"✓ Best reward: {best_reward:.2f}")

        algo.stop()

    elif args.mode == "test":
        if not args.checkpoint:
            print("Error: Provide --checkpoint")
            sys.exit(1)

        # algo = PPO.from_checkpoint(args.checkpoint)
        chkpt_path = os.path.abspath(args.checkpoint)
        algo = PPO.from_checkpoint(chkpt_path)
        
        # Ensure test config also uses the robust paths
        test_config = get_sim_config()
        test_config["gui"] = True # Enable GUI for test
        
        env = TrafficEnv(test_config)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            phase = "N-S" if action == 1 else "E-W"
            print(f"Step {step_count:3d} | Phase: {phase} | Col: {info['collisions']} | Rew: {reward:.2f}")

        env.close()
        algo.stop()

    ray.shutdown()