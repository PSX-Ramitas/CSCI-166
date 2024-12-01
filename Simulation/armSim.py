import pybullet as p
import math
import time
import numpy as np
import pybullet_data

class ArmEnv:
    def __init__(self, gui=True, mSteps=500):
        # initialize variables
        self.gui = gui
        self.physics_client = None
        self.planeId = None
        self.armId = None
        self.cubeId = None
        self.Tray = None
        self.stepSize = 0.00872665
        self.actionSpace = 729
        self.cubeStartPos = [0.25, 0.0, 0.05]
        self.trayStartPos = [0.0, 0.3, 0.01]
        self.endPoint = self.trayStartPos.copy()
        self.endPoint[2] += 0.2

        # set up variables for camera
        self.cameraYawId = 17
        self.cameraPitchId = 18
        self.cameraYaw = -0.3
        self.cameraPitch = -0.8

        # an array of servo Ids
        # goes from base upward
        # last two are right claw and left claw
        self.ServoIds = [2, 3, 4, 7, 9, 13, 15]

        # array for servo angles
        # corresponds to servo ids
        self.ServoAngles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # variables to keep track of updates
        self.reward = 0
        self.phase = 0
        self.state = None
        self.padTouch = False
        self.finished = False
        self.CubeTouchTray = False
        self.CubeTouchFloor = False
        self.steps = 0
        self.maxSteps = mSteps

    def connect(self):
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print("connected to pybullet simulation")

    def disconnect(self):
        # diconnect form the simulation
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            self.planeId = None
            self.armId = None
            print("diconnected from pybullet simulation")

    def load_environment(self):
        # Set the initial conditions of the simulation
        p.setGravity(0,0,-9.8)
        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0,0,0.01]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.armId = p.loadURDF("ArmObj/Robot_Arm.urdf", startPos, startOrientation, useFixedBase=True)
        self.cubeId = p.loadURDF("ArmObj/Cube.urdf", self.cubeStartPos)
        self.TrayId = p.loadURDF("ArmObj/Tray.urdf", self.trayStartPos)

        # Get the total number of joints (which is also the number of child links)
        num_joints = p.getNumJoints(self.armId)


        # Enable collisions between all links of the same object
        for joint_index in range(num_joints):
            # Setting collision filter to allow self-collision (i.e., link with itself)
            p.setCollisionFilterGroupMask(self.armId, joint_index, -1, -1)  # Enable self-collision for all links

        # Set the joint angle (position control mode)
        # move camera yaw to correct position
        p.setJointMotorControl2(
            self.armId,             # Robot ID
            self.cameraYawId,            # Joint index
            controlMode=p.POSITION_CONTROL,  # Position control mode
            targetPosition=self.cameraYaw,    # Target joint angle in radians
            force=500            # Maximum force to apply
        )

        # move camera pitch to correct position
        p.setJointMotorControl2(
            self.armId,             # Robot ID
            self.cameraPitchId,            # Joint index
            controlMode=p.POSITION_CONTROL,  # Position control mode
            targetPosition=self.cameraPitch,    # Target joint angle in radians
            force=500            # Maximum force to apply
        )

        for i in range(50):
            self.step()

    def resetSim(self):
        if self.physics_client is not None:
            p.resetSimulation()

            # array for servo angles
            # corresponds to servo ids
            self.ServoAngles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # variables to keep track of updates
            self.reward = 0
            self.phase = 0
            self.state = None
            self.padTouch = False
            self.finished = False
            self.CubeTouchTray = False
            self.CubeTouchFloor = False
            self.steps = 0

            self.load_environment()

            rgb_image = self.getCameraImage()

            # Convert the image from RGBA (4 channels) to RGB (3 channels)
            rgb_image = rgb_image[:, :, :3]

            self.state = [rgb_image, self.ServoAngles]

        return self.state

    def setMotorsPosition(self):
        # set default servo joint states

        for i in range(7):
            # move camera pitch to correct position
            p.setJointMotorControl2(
                self.armId,             # Robot ID
                self.ServoIds[i],            # Joint index
                controlMode=p.POSITION_CONTROL,  # Position control mode
                targetPosition=self.ServoAngles[i],    # Target joint angle in radians
                force=100,            # Maximum force to apply
                positionGain=1,  # Increase stiffness
                velocityGain=1   # Increase damping
            )

    def setMotorVelocity(self):
        # set default servo joint states

        for i in range(7):
            # move camera pitch to correct position
            p.setJointMotorControl2(
                self.armId,             # Robot ID
                self.ServoIds[i],            # Joint index
                controlMode=p.VELOCITY_CONTROL,  # Position control mode
                targetVelocity=0,
                force=10,            # Maximum force to apply
            )
    
    def updateServo(self, action: int, servo: int):
        # if action is 1 then move by positive tenth of a degree
        if (action == 1):
            self.ServoAngles[servo] -= self.stepSize
            if (self.ServoAngles[servo] <= -1.57):  # check if went beyond limits
                self.reward -= 10
                return False
        
        # if action is 2 then move by a negative tenth of a degree
        elif (action == 2):
            self.ServoAngles[servo] += self.stepSize
            if (self.ServoAngles[servo] >= 1.57): #check if went beyond limits
                self.reward -= 10
                return False
            
        return True
    
    def updateClawsServo(self, action: int):
        # if action is 1 then move by positive tenth of a degree
        if (action == 1):
            self.ServoAngles[5] += self.stepSize
            self.ServoAngles[6] -= self.stepSize
            if (self.ServoAngles[5] >= 0):  # check if went beyond limits
                self.reward -= 10
                return False
        
        # if action is 2 then move by a negative tenth of a degree
        elif (action == 2):
            self.ServoAngles[5] -= self.stepSize
            self.ServoAngles[6] += self.stepSize
            if (self.ServoAngles[5] <= -0.785398): #check if went beyond limits
                self.reward -= 10
                return False
        
        return True
    
    def collisionCheck(self):
        contact_points = p.getContactPoints()
        self.padTouch = False

        for contact in contact_points:
            # if contact is with the ground
            if contact[1] == self.planeId:
                # if contact is with robot
                if contact[2] == self.armId:
                    # check that contact is not with the base
                    if contact[4] != -1:
                      self.reward -= 10
                      return False
              
                # check if contact is with the cube
                # also check what phase it is, only matter if in phase 3
                # to not let cube drag on floor
                elif contact[2] == self.cubeId:
                    self.CubeTouchFloor = True                   

            # check if it is the robot contacting something else
            # also contact will not be with ground since that would appear first
            elif contact[1] == self.armId:
                # make sure it is the cube that is being touched
                if contact[2] == self.cubeId:
                    # if the contact is not with the touch pad, if so return false and have penalty
                    if contact[3] != 14 and contact[3] != 15:
                        self.reward -= 10
                        return False
                    # if it is both touch pad touching the cube then set flag to true
                    elif contact[3] == 14 and contact[3] == 15:
                        self.padTouch = True
                else:
                    self.reward -= 10
                    return False
            
            elif contact[1] == self.cubeId:
                if contact[2] == self.TrayId:
                    if contact[4] == -1:
                        self.CubeTouchTray = True
                    
            
        return True


    def takeAction(self, action: int):
        self.reward = 0
        self.steps += 1

        prev_pos = self.ServoAngles.copy()
        control = self.convInttoBase3(action)
        take_step = True
        pos_ClawL = np.array(p.getLinkState(self.armId, 14)[0])
        pos_ClawR = np.array(p.getLinkState(self.armId, 15)[0])
        prev_cube = p.getBasePositionAndOrientation(self.cubeId)

        prev_pos_C = ((pos_ClawL - pos_ClawR) * 0.5) + pos_ClawR

        # loop through the servos updating positions
        for i in range(5):
            take_step = (take_step and self.updateServo(control[i], i))
        
        # update claw seperately since it is two joints in simulation but one servo in real world
        take_step = (take_step and self.updateClawsServo(control[5]))
        
        if take_step:
            # run simulation for a moment to let arm update itself
            # check for collisions then and penalise for them
            for i in range(5):
                self.step()
                take_step = (take_step and self.collisionCheck())
                if not take_step:
                    break
            
            # if there is a collision don't do this and simply back step

            if take_step:
                pos1 = np.array(p.getLinkState(self.armId, 14)[0])
                pos2 = np.array(p.getLinkState(self.armId, 15)[0])

                posC = ((pos1 - pos2) * 0.5) + pos2

                cube, _ = p.getBasePositionAndOrientation(self.cubeId)

                dist1 = math.dist(cube, prev_pos_C)
                dist2 = math.dist(cube, posC)

                if self.phase == 0:
                    prev_dist = abs(0.3 - prev_pos[6])
                    dist = abs(0.3 - self.ServoAngles[6])
                    if prev_dist > dist:
                        self.reward += 1
                    elif prev_dist < dist:
                        self.reward -= 1

                    if dist1 > dist2:
                        self.reward += 2
                    elif dist1 < dist2:
                        self.reward -= 2
                    
                    if dist2 < 0.02:
                        self.phase = 1
                        self.reward += 20

                elif self.phase == 1:
                    prev_dist = abs(0.3 - prev_pos[6])
                    dist = abs(0.3 - self.ServoAngles[6])
                    if prev_dist < dist:
                        self.reward += 1
                    elif prev_dist > dist:
                        self.reward -= 1

                    if dist1 > dist2:
                        self.reward += 1
                    elif dist1 < dist2:
                        self.reward -= 1
                    
                    if dist2 > 0.03:
                        self.phase = 1
                        self.reward -= 10
                    
                    if self.padTouch:
                        self.phase = 2

                elif self.phase == 2:

                    if dist2 > dist1:
                        self.reward -= 1
                    
                    if self.CubeTouchFloor:
                        self.reward -= 1
                    
                    if dist2 > 0.03:
                        self.phase = 1
                        self.reward -= 10

                    dist3 = math.dist(cube, self.endPoint)
                    if dist3 < 0.02:
                        self.reward += 20
                        self.phase = 3

                elif self.phase == 3:
                    prev_dist = abs(0.3 - prev_pos[6])
                    dist = abs(0.3 - self.ServoAngles[6])
                    if prev_dist > dist:
                        self.reward += 1
                    elif prev_dist < dist:
                        self.reward -= 1

                    if dist2 > 0.03 and self.CubeTouchFloor:
                        self.phase = 1
                        self.reward -= 10
                    
                    tray, _ = p.getBasePositionAndOrientation(self.TrayId)
                    dist4 = math.dist(cube, tray)
                    dist5 = math.dist(prev_cube, tray)
                    if dist4 < dist5:
                        self.reward += 1
                    
                    if self.CubeTouchTray:
                        self.reward += 100
                        self.finished = True

            else:
                self.ServoAngles = prev_pos

        else:
            self.ServoAngles = prev_pos

        self.CubeTouchFloor = False
        
        rgb_image = self.getCameraImage()

        # Convert the image from RGBA (4 channels) to RGB (3 channels)
        rgb_image = rgb_image[:, :, :3]
        
        observation = [rgb_image, self.ServoAngles]

        return observation, self.reward, self.finished


    def step(self):
        self.setMotorsPosition()
        p.stepSimulation()

    def getCameraImage(self):
        # get position and orientation of the link representing physical camera
        camera_position, camera_orientation,_,_,_,_  = p.getLinkState(self.armId, 19)

        # set camera offset so it is not in the link
        camera_offset = [0, 0.015, 0]

        # Calculate the camera's eye position by adding the offset to the robot's position
        eye_position = [camera_position[0] + camera_offset[0], 
                        camera_position[1] + camera_offset[1], 
                        camera_position[2] + camera_offset[2]]
        
        # increase the offset a bit so it is looking ahead
        camera_offset = [0, 1, 0]

        camera_orientation = p.getEulerFromQuaternion(camera_orientation)
        Roll  = camera_orientation[1] * (180.0 / math.pi)
        Pitch = camera_orientation[0] * (180.0 / math.pi)
        Yaw   = camera_orientation[2] * (180.0 / math.pi)
        
        # Compute the view matrix (camera's position and orientation)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(eye_position, 
                                                          distance=0.1, 
                                                          yaw=Yaw,
                                                          pitch=Pitch, 
                                                          roll=Roll,
                                                          upAxisIndex=2)
        
        # Define the projection matrix (field of view, aspect ratio, near/far planes)
        fov = 60  # Field of view in degrees
        aspect_ratio = 1.0  # Aspect ratio (width/height of the output image)
        near_plane = 0.1  # Near clipping plane distance
        far_plane = 100  # Far clipping plane distance

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near_plane, far_plane)

        # Capture the image from the camera's viewpoint
        width = 480
        height = 270
        image = p.getCameraImage(width, height, view_matrix, projection_matrix)

        return np.array(image[2])  # Return the RGB image (height x width x 4)
    
    def convInttoBase3(self, x: int):
        out = [0,0,0,0,0,0]
        if x > 0 and x < 730:
            j = 5
            for i in range(6):
                out[j] = x % 3
                x = x // 3
                j -= 1
        return out
    
    def setEndPoint(self):
        self.endPoint, _ = p.getBasePositionAndOrientation(self.TrayId)
        self.endPoint[2] += 0.2

    def __del__(self):
        # ensure that simulation is ended on death of object
        if self.physics_client is not None:
            p.disconnect()


"""
#testing will be removed on final implementation
env = ArmEnv()
env.connect()
env.load_environment()

# Get the total number of joints in the robot
num_links = p.getNumJoints(env.armId)

# Enable debugging visualization for collision shapes
#p.setPhysicsEngineParameter(enableConeFriction=1)  # Optional: Visualization for debugging

# After loading the URDF and before stepping the simulation, you can visualize the collision shapes:
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

# Get the number of joints
num_joints = p.getNumJoints(env.armId)
print(f"Number of joints: {num_joints}")

# Loop through all joints to print joint info
for joint in range(num_joints):
    joint_info = p.getJointInfo(env.armId, joint)
    print(f"\nJoint {joint}:")
    print(f"  Name: {joint_info[1].decode('utf-8')}")
    print(f"  Type: {joint_info[2]}")
    print(f"  Damping: {joint_info[6]}")
    print(f"  Friction: {joint_info[7]}")
    print(f"  Lower Limit: {joint_info[8]}")
    print(f"  Upper Limit: {joint_info[9]}")
    print(f"  Max Force: {joint_info[10]}")
    print(f"  Max Velocity: {joint_info[11]}")
    print(f"  Parent Link Index: {joint_info[16]}")

for i in range(50):

    observation, reward, terminated = env.takeAction(360)

    #print(f"observation shape: {observation[0].shape}, {observation[1].shape}")
    #print(f"reward: {reward}")
    #print(f"terminated: {terminated}")

    time.sleep(1./240.)

env.disconnect()

test = env.convInttoBase3(360)
print(test)
"""