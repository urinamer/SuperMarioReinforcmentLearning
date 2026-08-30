import numpy as np
import torch
class ReplayBuffer():
    def __init__(self,size = 500000,obs_shape = (4,84,84)):
        self.max_size = size
        self.min_size = 1000
        #using uint8(1 byte) for pixels to save on memory. If we used float32(4 bytes) we would waste memory
        # pixel values range from 0 to 255 so uint8 is perfect
        self.states = np.zeros((self.max_size,*obs_shape),dtype=np.uint8)
        self.rewards = np.zeros(size,dtype=np.float32)
        self.actions = np.zeros(size,dtype=np.int64)
        self.next_states = np.zeros((size,*obs_shape),dtype=np.uint8)
        self.dones = np.zeros(size,dtype=bool)
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
        if self.curr_size < self.max_size:
            self.curr_size += 1


    def has_min_experiences(self):
        return self.curr_size >= self.min_size


    def get_random_batch(self,batch_size):
        indices =  np.random.choice(np.arange(0,self.curr_size),batch_size,replace=False)
        #casting uint8 back to float32 so neural network will except them
        return (
            torch.tensor(self.states[indices],dtype=torch.float32)/255.0,
            torch.tensor(self.rewards[indices]),
            torch.tensor(self.actions[indices]),
            torch.tensor(self.next_states[indices],dtype=torch.float32) /255.0,
            torch.tensor(self.dones[indices])
        )

