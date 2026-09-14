
import torch.nn as nn
import torch.optim
import os
from MarioCNNPPO import MarioCNNPPO
from RollOutBuffer import RollOutBuffer
from utils import get_env
from utils import plot_training_data
from utils import obs_to_tensor
import time

env = get_env(False,'rgb_array')
obs_dim = env.observation_space.shape
action_dim = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#gives action dim of 7 because of 7 different actions it could pick,outputs probs for each one
ppo_model = MarioCNNPPO(n_actions=action_dim).to(device)
optimizer = torch.optim.Adam(ppo_model.parameters(), lr=3e-4)
critic_loss_fn = nn.HuberLoss()
entropy_c2 = 0.04

#gives action dim of 1 because can only choose one action at a time
buffer = RollOutBuffer(size=2048, obs_dim=obs_dim,action_dim=1)
#learning rate decay
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1)

#checkpoint dir and file for saving
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "ppo_checkpoint.pt")


#data for plotting
num_of_ppo_elements = 0
num_of_clipped = 0
actor_losses = []
critic_losses = []
entropys = []
rewards = []
clipped_fractions = []
# max_x_pos= [0.0]


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


#loading previous data from google drive
start_fresh = True
start_episode = 0
if os.path.exists(checkpoint_path) and not start_fresh:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ppo_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_episode = ckpt["episode"] + 1
    rewards = ckpt.get("rewards", rewards)
    clipped_fractions = ckpt.get("clip fractions",clipped_fractions)
    actor_losses = ckpt.get("actor losses", actor_losses)
    critic_losses = ckpt.get("critic losses",critic_losses)
    entropys = ckpt.get("entropy",entropys)
    print(f"Resumed from episode {start_episode}")
else:
    print("starting fresh training")



#training loop
epoches = 250
scheduler.total_iters = epoches
starting_time = time.time()
current_obs, info = env.reset()
total_steps = 0
for episode in range(start_episode,start_episode + epoches):
    total_rewards = 0
    sum_actor_loss = 0
    sum_critic_loss = 0
    sum_entropy = 0
    num_of_updates = 0
    num_of_steps = 0
    num_of_ppo_elements = 0
    num_of_clipped = 0

    #saving data to not lose it in a crash
    if episode % 20 == 0:
        print("saving data")
        torch.save({
            "model_state": ppo_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "episode": episode,
            "rewards": rewards,
            "clip fractions": clipped_fractions,
            "actor losses": actor_losses,
            "critic losses": critic_losses,
            "entropy":entropys,
        }, checkpoint_path)

    #collecting experiences
    for _ in range(2048):
        with torch.no_grad():

            value,logits = ppo_model(obs_to_tensor(current_obs))
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
    batch_size = 128
    for epoch in range(5):
        for obs, action, old_log_prob, value, advantage, target in buffer.get_batches(batch_size):
            #recalculate predictions with current network states
            new_value,logits = ppo_model(obs)
            new_dist = torch.distributions.Categorical(logits=logits)
            # print(f'actions: {action}')
            new_log_prob = new_dist.log_prob(action.squeeze(-1))
            # losses
            actor_loss = ppo_loss(advantage, old_log_prob, new_log_prob)
            critic_loss = critic_loss_fn(new_value.squeeze(-1), target) #maybe add value clipping

            entropy = new_dist.entropy().mean()
            total_loss = actor_loss + critic_loss - (entropy_c2*entropy)

            sum_entropy += entropy.item()
            sum_critic_loss += critic_loss.item()
            sum_actor_loss += actor_loss.item()
            num_of_updates += 1


            # backprop
            ppo_model.zero_grad()
            total_loss.backward()
            optimizer.step()


    total_steps += buffer.size
    # decaying learning rate
    scheduler.step()
    rewards.append(total_rewards)
    entropys.append(sum_entropy/num_of_updates)
    critic_losses.append(sum_critic_loss/num_of_updates)
    actor_losses.append(sum_actor_loss / num_of_updates)
    clipped_fractions.append(num_of_clipped/num_of_ppo_elements)
    buffer.clear()

total_time = time.time()-starting_time
steps_per_second = total_steps/total_time
print(f"finished training in {total_time} seconds or {total_time/3600} hours")
print(f"{total_steps} steps in {total_time} seconds, = {steps_per_second} steps per second")
print(f"that means for 2M steps it would take {2_000_000/steps_per_second/3600} hours")
print(f"that means for 8M steps it would take {8_000_000/steps_per_second/3600} hours")

torch.save(ppo_model.state_dict(),"ppo_model_weights.pt")
plot_training_data([
    {"data": rewards, "title": "Total Reward per Episode", "ylabel": "Reward Score", "color": "green"},
    {"data": actor_losses, "title": "Average Actor Loss", "ylabel": "Average Loss", "color": "red"},
    {"data": critic_losses, "title": "Average Critic Loss", "ylabel": "Average Loss", "color": "yellow"},
    {"data": entropys, "title": "Average Entropy", "ylabel": "Average Entropy", "color": "blue"},
    {"data": clipped_fractions, "title": "Clip Fraction", "ylabel": "Percentage Clipped", "color": "orange"},
], save_path="graphs/ppo_training_graph.png")

env.close()

