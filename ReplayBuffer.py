import numpy as np
import torch
class ReplayBuffer():
    def __init__(self,size = 100000,obs_shape = (4,84,84)):
        self.max_size = size
        self.min_size = 1000
        self.states = np.zeros((self.max_size,*obs_shape),)
        self.rewards = np.zeros(size)
        self.actions = np.zeros(size)
        self.next_states = np.zeros((size,*obs_shape))
        self.dones = np.zeros(size)
        self.curr_index = 0
        self.curr_size = 0

    def store(self,state,reward,action,next_state,done):
        i = self.curr_index
        self.states[i] = state
        self.rewards[i] = reward
        self.actions[i] = action
        self.next_states[i] = next_state
        self.dones[i] = done

        self.curr_index = (self.curr_index+1) % self.max_size # loops around once reached the end
        if self.curr_size < self.min_size:
            self.curr_size += 1


    def has_min_experiences(self):
        return self.curr_size >= self.max_size


    def get_random_batch(self,batch_size):
        indices =  np.random.choice(np.arange(0,self.curr_size),batch_size,replace=False)

        return (
            torch.tensor(self.states[indices]),
            torch.tensor(self.rewards[indices]),
            torch.tensor(self.actions[indices]),
            torch.tensor(self.next_states[indices]),
            torch.tensor(self.dones[indices])
        )

