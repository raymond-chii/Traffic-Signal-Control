#!/usr/bin/env python3
import os
import sys
import traci
import numpy as np
import random

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

class RL:
    def __init__(self):
        # Simple 2 actions: 0 = North-South green, 1 = East-West green
        self.n_actions = 2
        
        # Q-table: state -> action values
        # discretize the state into bins
        self.q_table = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount = 0.9
        self.epsilon = 0.2  # exploration rate
        
    def get_state(self):
        # For simplicity, we'll use all vehicles on incoming edges
        
        # Get vehicle counts (this is simplified - in reality you'd check specific lanes)
        # Using edge IDs from your network
        north_count = len(traci.edge.getLastStepVehicleIDs("north_near_to_center"))
        south_count = len(traci.edge.getLastStepVehicleIDs("south_near_to_center"))
        east_count = len(traci.edge.getLastStepVehicleIDs("east_near_to_center"))
        west_count = len(traci.edge.getLastStepVehicleIDs("west_near_to_center"))
        
        def discretize(count):
            if count <= 2: return 0
            elif count <= 5: return 1
            elif count <= 8: return 2
            else: return 3
        
        state = (discretize(north_count), discretize(south_count), 
                discretize(east_count), discretize(west_count))
        
        return state
    
    def choose_action(self, state):

        # Initialize state in Q-table if not seen
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]  # 2 actions
        
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        # Exploit: best action
        return np.argmax(self.q_table[state])
    
    def get_reward(self):

        total_wait = 0
        vehicle_ids = traci.vehicle.getIDList()
        
        for vid in vehicle_ids:
            total_wait += traci.vehicle.getWaitingTime(vid)
        
        return -total_wait
    
    def update_q_table(self, state, action, reward, next_state):

        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def apply_action(self, action):
        # Apply action to traffic light
        # Action 0 = North-South green (GgGgrrrrGgGgrrrrrGrG)
        # Action 1 = East-West green (rrrrGgGgrrrrGgGgGrGr)

        if action == 0:
            # North-South green
            state = "GgGgrrrrGgGgrrrrrGrG"
        else:
            # East-West green
            state = "rrrrGgGgrrrrGgGgGrGr"
        
        traci.trafficlight.setRedYellowGreenState("center", state)

def train_rl_agent(episodes=50, steps_per_episode=600):

    agent = RL()
    episode_rewards = []
    
    for episode in range(episodes):
        # Start SUMO
        traci.start(["sumo", "-n", "simulation/network.net.xml", 
                     "-r", "simulation/routes.rou.xml"])
        
        total_reward = 0
        state = agent.get_state()
        
        for step in range(steps_per_episode):
            if step % 30 == 0:
                action = agent.choose_action(state)
                agent.apply_action(action)
            
            traci.simulationStep()
            
            # Get reward and next state every 30 seconds
            if step % 30 == 0 and step > 0:
                reward = agent.get_reward()
                next_state = agent.get_state()
                
                # Update Q-table
                agent.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
        
        episode_rewards.append(total_reward)
        traci.close()
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")
        
        # Reduce exploration over time
        agent.epsilon = max(0.05, agent.epsilon * 0.95)
    
    return agent, episode_rewards

def test_agent(agent, steps=600):

    traci.start(["sumo", "-n", "simulation/network.net.xml", 
                 "-r", "simulation/routes.rou.xml"])
    
    total_wait = 0
    state = agent.get_state()
    
    print("\nTesting trained agent...")
    for step in range(steps):
        if step % 30 == 0:
            action = agent.choose_action(state)
            agent.apply_action(action)
            state = agent.get_state()
            
            phase = "North-South" if action == 0 else "East-West"
            vehicles = len(traci.vehicle.getIDList())
            print(f"Step {step}: Phase={phase}, Vehicles={vehicles}, State={state}")
        
        traci.simulationStep()
        
        # Accumulate waiting time
        for vid in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(vid)
    
    traci.close()
    return total_wait

def test_fixed_time(steps=600):
    traci.start(["sumo", "-n", "simulation/network.net.xml", 
                 "-r", "simulation/routes.rou.xml"])
    
    total_wait = 0
    
    print("\nTesting fixed-time control...")
    for step in range(steps):
        # Simple fixed timing: 30s N-S, 30s E-W
        if (step // 30) % 2 == 0:
            traci.trafficlight.setRedYellowGreenState("center", "GgGgrrrrGgGgrrrrrGrG")
        else:
            traci.trafficlight.setRedYellowGreenState("center", "rrrrGgGgrrrrGgGgGrGr")
        
        traci.simulationStep()
        
        if step % 30 == 0:
            vehicles = len(traci.vehicle.getIDList())
            print(f"Step {step}: Vehicles={vehicles}")
        
        for vid in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(vid)
    
    traci.close()
    return total_wait

if __name__ == "__main__":
    
    # Train the agent
    print("\nTraining RL agent")
    agent, rewards = train_rl_agent(episodes=40, steps_per_episode=600)
    
    print(f"\nTraining complete!")
    print(f"Q-table size: {len(agent.q_table)} states learned")
    
    print("COMPARISON")
    print("="*50)
    
    rl_wait = test_agent(agent, steps=600)
    fixed_wait = test_fixed_time(steps=600)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"RL Agent Total Wait Time: {rl_wait:.2f} seconds")
    print(f"Fixed-Time Total Wait Time: {fixed_wait:.2f} seconds")
    print(f"Improvement: {((fixed_wait - rl_wait) / fixed_wait * 100):.1f}%")