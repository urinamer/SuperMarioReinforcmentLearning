import sys
import numpy as np
import torch.nn as nn
import torch.optim

from MarioCNNPPO import MarioCNNPPO

from RollOutBuffer import RollOutBuffer
from utils import get_env
from utils import plot_training_data

env = get_env(False)
obs_dim = env.observation_space.shape
action_dim = 7

#gives action dim of 7 because of 7 different actions it could pick,outputs probs for each one
ppo_model = MarioCNNPPO(n_actions=action_dim)
optimizer = torch.optim.Adam(ppo_model.parameters(), lr=3e-4)
critic_loss_fn = nn.HuberLoss()
#gives action dim of 1 because can only choose one action at a time
buffer = RollOutBuffer(size=2048, obs_dim=obs_dim,action_dim=1)
#learning rate decay
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=3000)


#data for plotting
num_of_ppo_elements = 0
num_of_clipped = 0
losses = [0.0]
rewards = [0]
clipped_fractions = [0.0]


#translates environment obs to tensors because env outputs lazyframes objects.
def obs_to_tensor(obs):
    arr = np.array(obs, dtype=np.float32)          # LazyFrames -> ndarray
    if arr.ndim == 4 and arr.shape[-1] == 1:        # squeeze stray channel dim if present
        arr = arr.squeeze(-1)
    return torch.tensor(arr).unsqueeze(0)           # (4, 84, 84) -> (1, 4, 84, 84)

def ppo_loss(advantage, old_log_prob, new_log_prob, clip_epsilon=0.2):
    global num_of_clipped
    global num_of_ppo_elements

    ratio = torch.exp(new_log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage

    with torch.no_grad():
        num_of_clipped += ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).sum().item()# gets number of times clipped
        num_of_ppo_elements += ratio.numel()
    return -torch.min(surr1, surr2).mean()

#training loop
current_obs, info = env.reset()
for episode in range(2):
    total_rewards = 0
    sum_actor_loss = 0
    num_of_steps = 1
    num_of_ppo_elements = 0
    num_of_clipped = 0
    #collecting experiences
    for _ in range(2048):
        with torch.no_grad():

            value,logits = ppo_model(   obs_to_tensor(current_obs))
            # print(f"logits: {logits}")
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            # print(f"action {action}")
            log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        buffer.store(obs_to_tensor(current_obs).numpy(), action.numpy(), log_prob.item(), reward, done, value.item())

        if done:
            current_obs, info = env.reset()
        else:
            current_obs = next_obs

        total_rewards += reward


    # estimating advantages with GAE
    with torch.no_grad():
        last_value,_ = ppo_model(obs_to_tensor(current_obs))
    buffer.calculateAdvantagesAndReturns(last_value, done)

    #advantage normalization
    buffer.normalize_advantages()


    #actual training
    batch_size = 100
    for epoch in range(5):
        for obs, action, old_log_prob, value, advantage, target in buffer.get_batches(batch_size):
            #recalculate predictions with current network states
            new_value,logits = ppo_model(obs)
            new_dist = torch.distributions.Categorical(logits=logits)
            print(f'actions: {action}')
            new_log_prob = new_dist.log_prob(action)

            # losses
            actor_loss = ppo_loss(advantage, old_log_prob, new_log_prob)
            critic_loss = critic_loss_fn(new_value, target)#maybe add value clipping

            total_loss = actor_loss + critic_loss

            sum_actor_loss += actor_loss.sum().item()
            num_of_steps += batch_size


            # backprop
            ppo_model.zero_grad()
            total_loss.backward()
            optimizer.step()

    rewards.append(total_rewards)
    losses.append(sum_actor_loss / num_of_steps)
    clipped_fractions.append(num_of_clipped/num_of_ppo_elements)
    buffer.clear()

torch.save(ppo_model.state_dict(),"ppo_model_weights.pt")
plot_training_data([
    {"data": rewards, "title": "Total Reward per Episode", "ylabel": "Reward Score", "color": "green"},
    {"data": losses, "title": "Average Actor Loss", "ylabel": "Average Loss", "color": "red"},
    {"data": clipped_fractions, "title": "Clip Fraction", "ylabel": "Percentage Clipped", "color": "orange"},
], save_path="graphs/ppo_training_graph.png")



env.close()

