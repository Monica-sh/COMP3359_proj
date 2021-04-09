import copy
import pdb
import numpy as np
from sklearn.preprocessing import normalize
import random
import torch


class ReplayMemory(object):
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros(capacity)
        self.reward = np.zeros(capacity)
        self.next_state = np.zeros((capacity, state_dim))
        self.position = 0
        self.count = 0

        self.state_dim = state_dim

    def reset(self):
        self.__init__(self.capacity, self.state_dim)

    def push(self, state, action, reward, next_state):
        """Saves a transition."""
        length_to_push = len(state)
        while length_to_push > 0:
            length = length_to_push if self.position + length_to_push < self.capacity else \
                self.position + length_to_push - self.capacity
            length_to_push = length_to_push - length

            self.state[self.position:self.position + length] = state[:length]
            self.action[self.position:self.position + length] = action[:length]
            self.reward[self.position:self.position + length] = reward[:length]
            self.next_state[self.position:self.position + length] = next_state[:length]
            self.position = (self.position + length) % self.capacity
            self.count += length

    def sample(self, batch_size):
        self.process_reward()
        state_batch, action_batch, reward_batch, next_state_batch = [], [], [], []

        for i in range(batch_size):
            idx = random.randint(0, len(self)-1)
            state_batch.append(self.state[idx])
            action_batch.append(self.action[idx])
            reward_batch.append(self.reward[idx])
            next_state_batch.append(self.next_state[idx])

        reward_batch = normalize(np.array(reward_batch).reshape(1, -1), norm='max')
        return torch.FloatTensor(state_batch), \
               torch.LongTensor(action_batch).unsqueeze(dim=1), \
               torch.FloatTensor(reward_batch), \
               torch.FloatTensor(next_state_batch)

    def __len__(self):
        return self.position if self.count < self.capacity else self.capacity

    def process_reward(self):
        none_idxes = np.where(np.isnan(self.reward))[0]
        for none_idx in none_idxes:
            day_count = 0
            for i in range(none_idx, len(self)):
                day_count += 1

                if not np.isnan(self.reward[i]):
                    profit = self.reward[i]
                    self.reward[none_idx:i] = [profit / day_count for _ in range(none_idx, i)]
                    break
