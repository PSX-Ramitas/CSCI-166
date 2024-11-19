from stable_baselines3 import PPO
from realSimPPO import ClawPickEnv

# Load the environment and model
env = ClawPickEnv()
model = PPO.load("claw_pick_model")

# Test the agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
