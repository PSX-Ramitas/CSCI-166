import pybullet as p
import pybullet_data
import time

# Start PyBullet in GUI mode
p.connect(p.GUI)

# Load the plane and set the gravity
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add default data path

# Load your URDF file
robot_id = p.loadURDF("C:/Users/Silver Surface Pro/CSCI-166-Project/CSCI-166/robot.urdf", basePosition=[0, 0, 0])


# Load a block to be picked up
block_id = p.loadURDF("cube.urdf", basePosition=[0.5, 0, 0])

# Simulation loop
while True:
    p.stepSimulation()
    
    # Example: Get the joint states and control logic here
    joint_states = p.getJointStates(robot_id, range(p.getNumJoints(robot_id)))
    
    # Example: Apply control commands to move the robot (this is a placeholder)
    # You need to define your control logic or reinforcement learning algorithm here
    
    time.sleep(1./240.)  # Maintain the simulation time step

# Disconnect from PyBullet
p.disconnect()
