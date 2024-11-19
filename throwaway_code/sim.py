import pybullet as p
import pybullet_data
import time

# Connect to the physics server
p.connect(p.GUI)

# Set the path to PyBullet's built-in assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane and set gravity
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)

# Load the robot
robot_id = p.loadURDF("robot.urdf", useFixedBase=True)
#tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
tableUid = p.loadURDF("table/table.urdf"),basePosition=[0.5,0,-0.65]

# Move each joint in the robot
num_joints = p.getNumJoints(robot_id)
for joint in range(num_joints):
    target_position = 0.1
   # p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=target_position)

# Run the simulation
for _ in range(100000):
    p.stepSimulation()
    time.sleep(1. / 240.)

# Disconnect from the server
p.disconnect()
