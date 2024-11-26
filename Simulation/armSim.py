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

        # set up variables for camera
        self.cameraYawId = 17
        self.cameraPitchId = 18
        self.cameraYaw = -0.5
        self.cameraPitch = -1

        # an array of servo Ids
        # goes from base upward
        # last two are right claw and left claw
        self.ServoIds = [2,3,4,7,9,11,12]

        # array for servo angles
        # corresponds to servo ids
        self.ServoAngles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
        self.armId = p.loadURDF("ArmObj/Robot_Arm.urdf", startPos, startOrientation)

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
        self.setMotorVelocity()
    
    def compensateGravity(self):
        for i in range(7):
            joint_state = p.getJointState(self.armId, self.ServoIds[i])
            joint_position = joint_state[0]

            # Calculate the mass of the link
            link_mass = p.getDynamicsInfo(self.armId, self.ServoIds[i])[0]  # Mass of the link

            # Calculate the torque required to counteract gravity
            # Using mass * gravity to find the gravitational force on the link
            gravity_torque = link_mass * 9.8  # Apply gravity force

            # Apply torque to each joint to compensate for gravity
            p.setJointMotorControl2(
                self.armId,
                self.ServoIds[i],
                p.TORQUE_CONTROL,
                force=gravity_torque
            )

    def setMotorsPosition(self):
        # set default servo joint states

        for i in range(7):
            # move camera pitch to correct position
            p.setJointMotorControl2(
                self.armId,             # Robot ID
                self.ServoIds[i],            # Joint index
                controlMode=p.POSITION_CONTROL,  # Position control mode
                targetPosition=self.ServoAngles[i],    # Target joint angle in radians
                force=10,            # Maximum force to apply
                positionGain=0.1,  # Increase stiffness
                velocityGain=0.1   # Increase damping
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
                force=100,            # Maximum force to apply
            )
    
    def updatePosition(self, action: int):
        i = 0

    def step(self):
        self.setMotorsPosition()
        p.stepSimulation()

    def getCameraImage(self):
        # get position and orientation of the link representing physical camera
        camera_position, camera_orientation,_,_,_,_  = p.getLinkState(self.armId, 19)

        # set camera offset so it is not in the link
        camera_offset = [0, 0.001, 0]

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

        print(camera_position)
        
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


for i in range(1000):
    
    rgb_image = env.getCameraImage()

    # Convert the image from RGBA (4 channels) to RGB (3 channels)
    rgb_image = rgb_image[:, :, :3]
    
    # Display the captured image
    #plt.imshow(rgb_image)
    #plt.title("Camera View from Link's Perspective")
    #plt.show()

    env.step()
    #time.sleep(1./240.)

env.disconnect()

test = env.convInttoBase3(728)
print(test)