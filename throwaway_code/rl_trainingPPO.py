from stable_baselines3 import PPO
from realSimPPO import ClawPickEnv

env = ClawPickEnv()
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("claw_pick_model")
