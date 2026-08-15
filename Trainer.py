import sys
from importlib.metadata import pass_none

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

#GAE
def calculate_advantages():


def get_mini_batches():
    pass

def calculateCriticLoss():
    pass

def calculateActorLoss():
    pass

advantages = np.zeros(1024)
values = np.zeros(1024)
dones = np.zeros(1024)# saves for GAE calculations
rewards = np.zeros(1024)
probs = np.zeros(1024)
states = np.zeros(1024)# why need to save states

epochs = 1000
for i in range(epochs):
    done = False
    state, info = env.reset()# state gives 84*84 pixel image
    #collect experiences
    for i in range(2048):#why 2048
        value,dist = ppo_model(state)
        action = torch.distributions.Categorical(dist).sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        values[i] = value
        dones[i] = done
        probs[i] = dist.log_prob(action)#why log value
        states[i] = state
        rewards[i] = reward


    for batch in get_mini_batches():
        for j in range(5):
            criticLoss = calculateCriticLoss()
            actorLoss = calculateActorLoss()
            optimizer.zero_grad()
            loss = None
            loss.backward()
            optimizer.step()




env.close()

