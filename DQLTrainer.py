import random

import torch
import torch.nn as nn
from utils import get_env
from utils import obs_to_tensor
from MarioCNNDQL import MarioCNNDQL
from ReplayBuffer import ReplayBuffer

env = get_env(False)
action_dim = 7
obs_dim = env.observation_space.shape[0]

DQLModel = MarioCNNDQL(n_actions=action_dim)
target_network = MarioCNNDQL(n_actions=action_dim)
target_network.load_state_dict(DQLModel.state_dict())#copy weights

optimizer = torch.optim.Adam(DQLModel.parameters(),lr=1e-4)
loss_fn = nn.HuberLoss()
buffer = ReplayBuffer()
#maybe add learning rate decay. not sure if it helps in DQL

max_steps_per_episode = 1000
epsilon = 1
epsilon_decay = 0.9998
target_network_update = 10000
discount_factor = 0.99

def choose_explore_exploit(actions,epsilon):
    num = random.random()
    if num > epsilon:
        return torch.argmax(actions).item(),torch.max(actions).item()
    else:
        index = random.randint(0,actions.shape[-1]-1)
        return index,actions[0,index]



total_steps = 0
for episode in range(1000):
    curr_state, info = env.reset()
    done = False
    episodes_steps = 0
    while not done and episodes_steps < max_steps_per_episode:
        total_steps += 1
        episodes_steps += 1
        with torch.no_grad():
            action,q_value = choose_explore_exploit(DQLModel(obs_to_tensor(curr_state)),epsilon)
            epsilon = max(epsilon*epsilon_decay,0.01)

            next_state,reward,terminated, truncated,info = env.step(action)
            done = terminated or truncated
            buffer.store(curr_state,reward, action, next_state,done)#storing in replay buffer
            curr_state = next_state

        if buffer.has_min_experiences():
            states,rewards,actions,next_states,dones = buffer.get_random_batch(batch_size=1000)# dont know what batch size to put
            q_values = DQLModel(states).gather(1, actions.long().unsqueeze(1)).squeeze(1) #Dont understand this line

            #calculating target q values using bellman equation
            with torch.no_grad():
                target_q_values = rewards + (~ dones).float() * discount_factor * torch.max(target_network(next_states),dim=1)[0]

            loss = loss_fn(q_values,target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_steps % target_network_update == 0:
            target_network.load_state_dict(DQLModel.state_dict())  # copy weights,update target network


