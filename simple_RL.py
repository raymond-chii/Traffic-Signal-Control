#!/usr/bin/env python3
import os
import sys
import traci
import numpy as np
import random

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# Define SUMO command arguments for consistency
SUMO_CMD = ["sumo",
            "-n", "simulation/network.net.xml",
            "-r", "simulation/routes.rou.xml",
            "--no-warnings", "true"]

class RL:
    def __init__(self):
        self.n_actions = 2
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.9
        self.epsilon = 1.0
        
    def get_state(self):
        north_count = traci.edge.getLastStepHaltingNumber("north_near_to_center")
        south_count = traci.edge.getLastStepHaltingNumber("south_near_to_center")
        east_count = traci.edge.getLastStepHaltingNumber("east_near_to_center")
        west_count = traci.edge.getLastStepHaltingNumber("west_near_to_center")

        def discretize(count):
            if count <= 1: return 0
            elif count <= 3: return 1
            elif count <= 5: return 2
            else: return 3
        
        state = (discretize(north_count), discretize(south_count), 
                discretize(east_count), discretize(west_count))
        return state
    
    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        
        if random.random() < self.epsilon:
            return random.randint(0, 1) # Explore
        
        return np.argmax(self.q_table[state]) # Exploit
    
    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def apply_action(self, action):
            if action == 0:
                traci.trafficlight.setPhase("center", 0)
            else:
                traci.trafficlight.setPhase("center", 5)

# NEW HELPER FUNCTION
def get_wait_time_for_step():
    """Gets the total waiting time for all 4 incoming edges IN THE LAST SECOND."""
    north_wait = traci.edge.getWaitingTime("north_near_to_center")
    south_wait = traci.edge.getWaitingTime("south_near_to_center")
    east_wait = traci.edge.getWaitingTime("east_near_to_center")
    west_wait = traci.edge.getWaitingTime("west_near_to_center")
    return north_wait + south_wait + east_wait + west_wait

def train_rl_agent(episodes=50, steps_per_episode=600):
    agent = RL()
    
    for episode in range(episodes):
        traci.start(SUMO_CMD)
        
        state = agent.get_state()
        action = agent.choose_action(state)
        agent.apply_action(action)
        
        current_interval_wait = 0.0
        total_episode_reward = 0.0
        
        for step in range(steps_per_episode):
            traci.simulationStep()
            
            current_interval_wait += get_wait_time_for_step()
            
            if (step + 1) % 15 == 0:
                reward = -current_interval_wait
                total_episode_reward += reward
                
                next_state = agent.get_state()
                agent.update_q_table(state, action, reward, next_state)
                
                next_action = agent.choose_action(next_state)
                agent.apply_action(next_action)
                
                state = next_state
                action = next_action
                current_interval_wait = 0.0

        traci.close()
        
        print(f"Episode {episode+1: >3}/{episodes} | Reward: {total_episode_reward: >8.2f} | Epsilon: {agent.epsilon:.3f} | States: {len(agent.q_table)}")
        
        agent.epsilon = max(0.05, agent.epsilon * 0.95)
    
    return agent

def test_agent(agent, steps=600):
    agent.epsilon = 0.0
    traci.start(SUMO_CMD)
    
    state = agent.get_state()
    total_simulation_wait = 0.0
    
    print("\nTesting trained agent...")
    for step in range(steps):
        if step % 30 == 0:
            action = agent.choose_action(state)
            agent.apply_action(action)
            state = agent.get_state()
            
            phase = "North-South" if action == 0 else "East-West"
            print(f"Step {step: >3}: Phase={phase: <11} | State={state}")
        
        traci.simulationStep() 
        total_simulation_wait += get_wait_time_for_step()
    
    traci.close()
    return total_simulation_wait # Return the true total

def test_fixed_time(steps=600):
    traci.start(SUMO_CMD)
    total_simulation_wait = 0.0
    
    print("\nTesting fixed-time control (using default .net.xml program)...")
    for step in range(steps):
        traci.simulationStep()
        
        total_simulation_wait += get_wait_time_for_step()

    traci.close() 
    return total_simulation_wait # Return the true total

if __name__ == "__main__":
    
    TRAIN_EPISODES = 200
    print(f"\nTraining RL agent for {TRAIN_EPISODES} episodes...")
    agent = train_rl_agent(episodes=TRAIN_EPISODES, steps_per_episode=600)
    
    print(f"\nTraining complete!")
    print(f"Q-table size: {len(agent.q_table)} states learned")
    
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    
    rl_wait = test_agent(agent, steps=600)
    fixed_wait = test_fixed_time(steps=600)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"RL Agent Total Wait Time: {rl_wait:.2f} seconds")
    print(f"Fixed-Time Total Wait Time: {fixed_wait:.2f} seconds")
    
    improvement = ((fixed_wait - rl_wait) / fixed_wait * 100) if fixed_wait > 0 else 0
    print(f"Improvement: {improvement:.1f}%")