import pybullet as p
import math
import time
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt

class ArmEnv:
    def __init__(self, gui=True):
        # initialize variables
        self.gui = gui
        self.physics_client = None
        self.planeId = None
        self.armId = None
        self.cubeId = None
        self.stepSize = 0.00872665
        self.state = None
        self.padTouch = False
        self.finished = True

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
        self.ServoAngles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]

        # variables to keep track of updates
        self.reward = 0
        self.phase = 0
        self.current_state = None

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

    def reset(self):
        if self.physics_client is not None:
            p.resetSimulation()
            print("Simulation Reset")

    def load_environment(self):
        # Set the initial conditions of the simulation
        p.setGravity(0,0,-9.8)
        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0,0,0.01]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.armId = p.loadURDF("ArmObj/Robot_Arm.urdf", startPos, startOrientation, useFixedBase=True)
        self.cubeId = p.loadURDF("ArmObj/Cube.urdf", [0.4,0.0,0.05])

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

        self.setMotorsPosition()

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
            self.ServoIds[servo] += self.stepSize
            if (self.ServoAngles[servo] >= 1.57): #check if went beyond limits
                self.reward -= 10
                return False
            
        return True
    
    def updateClawsServo(self, action: int):
        # if action is 1 then move by positive tenth of a degree
        if (action == 1):
            self.ServoAngles[5] -= self.stepSize
            self.ServoAngles[6] += self.stepSize
            if (self.ServoAngles[5] <= 0.0):  # check if went beyond limits
                self.reward -= 10
                return False
        
        # if action is 2 then move by a negative tenth of a degree
        elif (action == 2):
            self.ServoAngles[5] += self.stepSize
            self.ServoAngles[6] -= self.stepSize
            if (self.ServoAngles[5] >= 0.785398): #check if went beyond limits
                self.reward -= 10
                return False
        
        return True
    
    def collisionCheck(self):
        contact_points = p.getContactPoints()
        self.padTouch = False

        for contact in contact_points:
            if contact[1] == 0:
                if contact[4] != -1:
                    self.reward -= 10
                    return False
            elif contact[1] == 1:
                if contact[3] != 14 and contact[3] != 15:
                    self.reward -= 10
                    return False
                elif contact[3] == 14 and contact[3] == 15:
                    self.padTouch = True
            
        return True


    def takeAction(self, action: int):
        self.reward = 0

        prev_pos = self.ServoAngles
        control = self.convInttoBase3(action)
        take_step = True
        pos_ClawL = p.getLinkState(self.armId, 14)[0]
        pos_ClawR = p.getLinkState(self.armId, 15)[0]

        prev_pos_C = ((pos_ClawL - pos_ClawR) * 0.5) + pos_ClawL

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

            if take_step:
                if self.phase == 0:
                    prev_dist = 0.3 - prev_pos[6]
                    dist = 0.3 - self.ServoAngles[6]
                    if prev_dist > dist:
                        self.reward += 1
                    
                    pos1 = p.getLinkState(self.armId, 14)[0]
                    pos2 = p.getLinkState(self.armId, 15)[0]

                    posC = ((pos1 - pos2) * 0.5) + pos1

                    cube = p.getLinkState(self.cubeId, -1)

                    dist1 = math.dist(cube, prev_pos_C)
                    dist2 = math.dist(cube, posC)

                    if dist1 > dist2:
                        self.reward += 2
                    
                    if dist2 < 0.01:
                        self.phase = 1

                elif self.phase == 1:
                    prev_dist = 0.3 - prev_pos[6]
                    dist = 0.3 - self.ServoAngles[6]
                    if prev_dist < dist:
                        self.reward += 1
                    
                    pos1 = p.getLinkState(self.armId, 14)[0]
                    pos2 = p.getLinkState(self.armId, 15)[0]

                    posC = ((pos1 - pos2) * 0.5) + pos1

                    cube = p.getLinkState(self.cubeId, -1)

                    dist1 = math.dist(cube, prev_pos_C)
                    dist2 = math.dist(cube, posC)

                    if dist1 > dist2:
                        self.reward -= 1
                    
                    if self.padTouch:
                        self.phase = 2

                elif self.phase == 2:
                    i = 0
                elif self.phase == 3:
                    i = 0

        else:
            self.ServoAngles = prev_pos

        
        return 


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
    
    def convInttoBase3(x: int):
        out = [0,0,0,0,0,0]
        if x > 0 and x < 730:
            j = 5
            for i in range(6):
                out[j] = x % 3
                x = x // 3
                j -= 1
        return out

    def __del__(self):
        # ensure that simulation is ended on death of object
        if self.physics_client is not None:
            p.disconnect()


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

for i in range(1000):
    
    rgb_image = env.getCameraImage()

    # Convert the image from RGBA (4 channels) to RGB (3 channels)
    rgb_image = rgb_image[:, :, :3]
    
    # Display the captured image
    #plt.imshow(rgb_image)
    #plt.title("Camera View from Link's Perspective")
    #plt.show()

    env.step()

    contact_points = p.getContactPoints(env.armId, env.armId)
    if len(contact_points) > 0:
        print(f"Number of Contact Points: {len(contact_points)}")
        for contact in contact_points:
            print(f"Contact Between point {contact[4]} and {contact[3]}")

    #time.sleep(1./240.)

env.disconnect()

test = env.convInttoBase3(728)
print(test)