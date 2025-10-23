import os
import sys
import traci

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

traci.start(["sumo", "-n", "simulation/network.net.xml", "-r", "simulation/routes.rou.xml"])

# print("Manually controlling traffic light...\n")


NS_GREEN = "GgGgrrrrGgGgrrrrrGrG"
# East-West green  
EW_GREEN = "rrrrGgGgrrrrGgGgGrGr"
# All red
ALL_RED = "rrrrrrrrrrrrrrrrrrrr"

for step in range(300):
    
    if (step // 30) % 2 == 0:
        traci.trafficlight.setRedYellowGreenState("center", NS_GREEN)
        phase = "North-South GREEN"
    else:
        traci.trafficlight.setRedYellowGreenState("center", EW_GREEN)
        phase = "East-West GREEN"
    
    traci.simulationStep()
    
    if step % 30 == 0:
        vehicles = traci.vehicle.getIDList()
        print(f"Time: {step}s | Phase: {phase} | Vehicles: {len(vehicles)}")

traci.close()
print("\nDone!")