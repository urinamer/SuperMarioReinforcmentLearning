import random

import torch
import torch.nn as nn
from utils import get_env, plot_training_data
from utils import obs_to_tensor
from MarioCNNDQL import MarioCNNDQL
from ReplayBuffer import ReplayBuffer

env = get_env(False)
action_dim = 7
obs_dim = env.observation_space.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DQLModel = MarioCNNDQL(n_actions=action_dim).to(device)
target_network = MarioCNNDQL(n_actions=action_dim).to(device)
target_network.load_state_dict(DQLModel.state_dict())#copy weights

optimizer = torch.optim.Adam(DQLModel.parameters(),lr=1e-4)
loss_fn = nn.HuberLoss()
buffer = ReplayBuffer()
#maybe add learning rate decay. not sure if it helps in DQL

max_steps_per_episode = 500
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

losses = [0.0]
q_values_list = [0.0]
rewards_list = [0]
epsilons = [epsilon]



total_steps = 0
for episode in range(2):
    print(episode)
    curr_state, info = env.reset()
    total_loss = 0
    total_rewards = 0
    total_q_values = 0
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
            buffer.store(obs_to_tensor(curr_state),reward, action, obs_to_tensor(next_state),done)#storing in replay buffer
            curr_state = next_state

        if buffer.has_min_experiences():
            batch_size = 1000
            states,rewards,actions,next_states,dones = buffer.get_random_batch(batch_size=batch_size)# dont know what batch size to put
            q_values = DQLModel(states).gather(1, actions.long().unsqueeze(1)).squeeze(1) #Dont understand this line

            total_q_values += q_values.sum().item()
            total_rewards += rewards.sum().item()
            #calculating target q values using bellman equation
            with torch.no_grad():
                target_q_values = rewards + (~ dones).float() * discount_factor * torch.max(target_network(next_states),dim=1)[0]

            loss = loss_fn(q_values,target_q_values)
            total_loss += loss.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_steps % target_network_update == 0:
            target_network.load_state_dict(DQLModel.state_dict())  # copy weights,update target network

    losses.append(total_loss / episodes_steps)
    q_values_list.append(total_q_values/episodes_steps)
    rewards_list.append(total_rewards)
    epsilons.append(epsilon)

torch.save(DQLModel.state_dict(),"DQL_model_weights.pt")
plot_training_data([
    {"data": rewards_list, "title": "Total Reward per Episode", "ylabel": "Reward Score", "color": "green"},
    {"data": losses, "title": "Average Actor Loss", "ylabel": "Average Loss", "color": "red"},
    {"data":q_values_list,"title":"Average Q value per episode","ylabel":"Average Q Value","color":"blue"},
    {"data": epsilons, "title": "Exploration Epsilon", "ylabel": "Epsilon", "color": "orange"},
], save_path="graphs/dql_training_graph.png")
