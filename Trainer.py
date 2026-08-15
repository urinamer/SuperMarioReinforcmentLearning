import sys
import numpy as np
import torch.nn as nn
import torch.optim
from MarioCNNPPO import MarioCNNPPO
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from shimmy import GymV21CompatibilityV0

# gym_super_mario_bros only registers with the OLD `gym` package, not
# gymnasium — so we build it with the legacy gym API first
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#then wrap it so it behaves like a gymnasium env
env = GymV21CompatibilityV0(env=env, render_mode="human")

ppo_model = MarioCNNPPO(env.action_space.shape)
optimizer = torch.optim.Adam(ppo_model.parameters(),lr=0.01)

def get_action():
    pass

#GAE
def calculate_advantages():
    pass


epochs = 1000
for i in range(epochs):
    done = False
    state, info = env.reset()
    while not done:
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        env.render()

env.close()

