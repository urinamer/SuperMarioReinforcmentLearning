import numpy as np
import torch


class RollOutBuffer:
    def __init__(self,size,obs_dim, action_dim):
        self.size = size
        self.current_insert = 0
        self.observations = np.zeros((size, *obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.old_log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.masks = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)

        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32) #the targets for the critic value

    def store(self,observation,action,log_prob,reward,mask,state_value):
        self.observations[self.current_insert] = observation
        self.actions[self.current_insert] = action
        self.old_log_probs[self.current_insert] = log_prob
        self.rewards[self.current_insert] = reward
        self.masks[self.current_insert] = mask
        self.values[self.current_insert] = state_value
        self.current_insert += 1


    #runs when buffer is full is full
    def calculateAdvantagesAndReturns(self,next_value,done,gamma = 0.99,lamda = 0.95):
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size-1:
                next_state_not_done = not done
            else:
                next_state_not_done = not self.masks[t+1]

            delta = self.rewards[t] + gamma*next_value * next_state_not_done  - self.values[t]
            gae = delta + lamda*gamma * gae*next_state_not_done

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

            next_value = self.values[t]


    def clear(self):
        self.current_insert = 0

    def get_batches(self,batch_size):
        indexes = np.arange(self.size)# making an array of 1,2,3,4,5...size-1
        np.random.shuffle(indexes)

        for start_index in range(0,self.size,batch_size):
            end_index = start_index + batch_size
            batch_slice = indexes[start_index:end_index]

            yield (
                torch.FloatTensor(self.observations[batch_slice]),#converting from numpy arrays to tensors
                torch.FloatTensor(self.actions[batch_slice]),
                torch.FloatTensor(self.old_log_probs[batch_slice]),
                torch.FloatTensor(self.values[batch_slice]),
                torch.FloatTensor(self.advantages[batch_slice]),
                torch.FloatTensor(self.returns[batch_slice])
            )



    def is_full(self):
        return self.size is self.current_insert

    def normalize_advantages(self):
        mean = np.mean(self.advantages)
        std = np.std(self.advantages)
        self.advantages = (self.advantages - mean)/(std+1e-8)



