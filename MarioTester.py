import torch

from MarioCNNPPO import MarioCNNPPO
from utils import get_env
from utils import obs_to_tensor


env = get_env(False,'human')
num_of_actions = 7
model = MarioCNNPPO(n_actions=num_of_actions)
#loading weights
state_dict = torch.load('ppo_model_weights.pt',weights_only=True)
model.load_state_dict(state_dict)


episodes = 3
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    num_of_steps = 0
    while(not done or num_of_steps <= 500):
        num_of_steps += 1
        value,logits = model(obs_to_tensor(obs))
        action = torch.distributions.Categorical(logits).sample()

        obs,reward,terminated,truncated,info = env.step(action.item())
        done = terminated or truncated

        env.render()




