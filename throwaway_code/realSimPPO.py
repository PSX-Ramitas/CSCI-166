import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces





class ClawPickEnv(gym.Env):
    def __init__(self):
        super(ClawPickEnv, self).__init__()
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load the robot and block
        self.robot_id = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.5])
        self.block_id = p.loadURDF("block.urdf", basePosition=[0.5, 0, 0.5])

        # Define observation and action spaces
        camera_obs_size = 64 * 64 * 3 
        lidar_obs_size = 1 
        observation_shape = (camera_obs_size + lidar_obs_size,)
        
        action_shape= (3,)
        
        self.observation_space = spaces.Box(low=-1, high=1, shape= observation_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape= action_shape, dtype=np.float32)

    def reset(self):
        # Reset the environment and return the initial observation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.robot_id = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.5])
        self.block_id = p.loadURDF("block.urdf", basePosition=[0.5, 0, 0.5])
        return self.get_observation()
    def check_observation_validity(self, rgb_img, lidar_data):
    # Check if the camera image is empty (all pixels are zero)
        if np.all(rgb_img == 0):
            return False  # Bad observation (empty image)

    # Check for 'inf' values in lidar data
        if np.any(np.isinf(lidar_data)):  # Check if lidar data contains 'inf' values
            return False  # Bad observation (lidar data contains 'inf')
    
    # Check if lidar data has NaN values (invalid measurements)
        if np.any(np.isnan(lidar_data)):
            return False  # Bad observation (lidar data contains NaNs)

    # Check if lidar data has unrealistic distances (e.g., too far or too close)
        if np.any(lidar_data < 0.1) or np.any(lidar_data > 50.0):  # Example threshold (adjust as necessary)
            return False  # Bad observation (unrealistic lidar readings)

        return True  # Good observation (if no issues found)

    def get_observation(self):
        # Capture images and lidar data
        rgb_img = self.get_camera_image()
        lidar_data = self.get_single_lidar_reading()
        # Check the validity of the observation
        if not self.check_observation_validity(rgb_img, lidar_data):
            return np.zeros_like(lidar_data)  # Return an indication of a bad observation
    
        # Replace 'inf' with a large number or a default value (if you want to sanitize the lidar data)
        lidar_data = np.where(np.isinf(lidar_data), 100.0, lidar_data)  # Replace inf with a max distance, e.g., 100 meters
    
        # Flatten the camera image and concatenate with lidar data to form the observation
        
        obs = np.concatenate([rgb_img.flatten(), lidar_data])
        print("OBSERVATION: ", obs)
        return obs

    def step(self, action):
        # Execute action and compute reward
        self.apply_action(action)
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        done = self.is_done()
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass  # Implement if needed

    def apply_action(self, action):
        # Apply the action to the robotic claw
        pass

    def compute_reward(self, observation):
    # Example of invalid state or failure
        if observation == "some_invalid_state":
            return -1.0  # Negative reward for invalid state (e.g., falling, going out of bounds)

        # Example of successful task completion (e.g., picking up a block)
        elif observation == "task_completed":
            return 10.0  # Large positive reward for successful task

    # Example of a closer state to the goal (positive feedback)
        elif observation == "close_to_target":
            return 1.0  # Small positive reward for getting closer to the goal

        # Example of a farther state from the goal (negative feedback)
        elif observation == "far_from_target":
            return -0.5  # Small negative reward for being far from the goal

    # Default reward (neutral state)
        else:
            return 0.0  # No reward for neutral states, agent didn't make progress
  
    def is_done(self):
        # Define terminal condition
        pass
    def get_camera_image(self):
        # Define camera position and orientation
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1, 1, 2],  # Camera position in space
            cameraTargetPosition=[0, 0, 1],  # Where the camera is looking at
            cameraUpVector=[0, 0,1] , # Camera's "up" direction
        )
        # Define the projection matrix for camera perspective
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=45.0,  # Field of view
            aspect=1.0,  # Aspect ratio (adjust based on your simulation view)
            nearVal=0.1,  # Near clipping plane
            farVal=3.1  # Far clipping plane
        )

        # Get camera image from the simulation
        width, height, rgb_img, _, _ = p.getCameraImage(
        width=64, height=64,  # Image resolution
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
        )
    
    # Reshape the returned image to a 3D numpy array (height, width, channels)
        rgb_img = np.reshape(rgb_img, (height, width, 4))  # Include alpha channel (RGBA)
        rgb_img = rgb_img[:, :, :3]  # Extract RGB channels (ignore alpha channel)
    
        return rgb_img

    def get_single_lidar_reading(self):
    # Simulate a LiDAR sensor by ray-casting around the robot
       # Get a single LiDAR reading directly in front of the claw
        lidar_origin = p.getBasePositionAndOrientation(self.robot_id)[0]
        lidar_direction = [0, 1, 0]  # Direction directly in front of the claw
        ray_end = [lidar_origin[0] + lidar_direction[0], lidar_origin[1] + lidar_direction[1], lidar_origin[2]]
        
        ray_results = p.rayTest(lidar_origin, ray_end)
        if ray_results[0][0] != -1:
            distance = ray_results[0][2]
        else:
            distance = float('inf')  # No hit detected
        
        return np.array([distance])