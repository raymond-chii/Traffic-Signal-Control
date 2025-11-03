#!/usr/bin/env python3
import numpy as np
import traci


class Grid:
    def __init__(self, grid_size=(8, 8), bounds=None):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        if bounds is None:
            self.bounds = (120, 120, 280, 280)
        else:
            self.bounds = bounds
            
        min_x, min_y, max_x, max_y = self.bounds
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.cell_width = self.width / self.cols
        self.cell_height = self.height / self.rows
        
    def get_cell(self, x, y):
        min_x, min_y, max_x, max_y = self.bounds
        
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return None
            
        col = int((x - min_x) / self.cell_width)
        row = int((y - min_y) / self.cell_height)
        
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        
        return (row, col)
    
    def get_vehicle_grid(self):

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                cell = self.get_cell(x, y)
                if cell is not None:
                    row, col = cell
                    grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
                
        return grid
    
    def get_pedestrian_grid(self):

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        person_ids = traci.person.getIDList()
        
        for person_id in person_ids:
            try:
                x, y = traci.person.getPosition(person_id)
                cell = self.get_cell(x, y)
                if cell is not None:
                    row, col = cell
                    grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
                
        return grid
    
    def get_speed_grid(self):
        """
        Returns a grid matrix with average vehicle speeds in each cell.
        
        Returns:
            numpy array of shape (rows, cols) with average speeds (m/s)
        """
        speed_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        count_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                cell = self.get_cell(x, y)
                if cell is not None:
                    row, col = cell
                    speed_grid[row, col] += speed
                    count_grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
        
        # Calculate averages (avoid division by zero)
        mask = count_grid > 0
        speed_grid[mask] = speed_grid[mask] / count_grid[mask]
        
        return speed_grid
    
    def get_waiting_grid(self):
        """
        Returns a grid matrix with counts of waiting vehicles (speed < 0.1 m/s).
        
        Returns:
            numpy array of shape (rows, cols) with waiting vehicle counts
        """
        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                
                if speed < 0.1:  # Vehicle is essentially stopped
                    cell = self.get_cell(x, y)
                    if cell is not None:
                        row, col = cell
                        grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
                
        return grid
    
    def get_full_observation(self):
        """
        Returns a dictionary with all grid observations.
        
        Returns:
            dict with keys: 'vehicles', 'pedestrians', 'speeds', 'waiting'
        """
        return {
            'vehicles': self.get_vehicle_grid(),
            'pedestrians': self.get_pedestrian_grid(),
            'speeds': self.get_speed_grid(),
            'waiting': self.get_waiting_grid()
        }
    
    def get_stacked_observation(self):
        """
        Returns all observations stacked as a 3D array for neural networks.
        
        Returns:
            numpy array of shape (4, rows, cols) with channels:
            0: vehicle counts
            1: pedestrian counts  
            2: average speeds
            3: waiting vehicle counts
        """
        obs = self.get_full_observation()
        return np.stack([
            obs['vehicles'],
            obs['pedestrians'],
            obs['speeds'],
            obs['waiting']
        ], axis=0)
    
    def print_grid(self, grid, title="Grid"):
        """Pretty print a grid matrix."""
        print(f"\n{title}:")
        print("-" * (self.cols * 4 + 1))
        for row in grid:
            print("|" + "|".join(f"{val:3d}" for val in row) + "|")
        print("-" * (self.cols * 4 + 1))


if __name__ == "__main__":
    import os
    import sys
    
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    
    SUMO_CMD = ["sumo",
            "-n", "simulation/network.net.xml",
            "-r", "simulation/routes.rou.xml",
            "--no-warnings", "true"]
    
    # Start SUMO
    traci.start(SUMO_CMD)
    
    # Create grid observer
    observer = Grid(grid_size=(8, 8), bounds=(120, 120, 280, 280))
    
    print("Running simulation with grid observation...")
    for step in range(500):
        traci.simulationStep()
        
        if step % 50 == 0:
            print(f"\n{'='*60}")
            print(f"Step {step} ({step} seconds)")
            print('='*60)
            
            obs = observer.get_full_observation()
            
            # Print vehicle counts
            observer.print_grid(obs['vehicles'], "Vehicle Counts")
            print(f"Total vehicles: {obs['vehicles'].sum()}")
            
            # Print waiting vehicles
            observer.print_grid(obs['waiting'], "Waiting Vehicles")
            print(f"Total waiting: {obs['waiting'].sum()}")
    
    traci.close()
    print("\nSimulation complete!")