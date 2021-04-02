from namedlist import namedlist
import copy
import numpy as np
from sklearn.preprocessing import normalize
import random
import torch

Transition = namedlist('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        self.process_reward()
        processed_memory = self.normalize_reward()
        return random.sample(processed_memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def process_reward(self):
        """
        Since the reward calculation is delayed until the agent sells the stock,
        this function handles such calculation.

        If the agent is not currently holding stocks and does not buy any, reward = 0.
        The reward of each day the agent holding stocks is the total profit when it sells
        the stocks, averaged over the days it holds it.

        """

        r_list = [trans.reward for trans in self.memory]
        first_none_idx = 0
        if None in r_list:
            first_none_idx = r_list.index(None)
        while True:
            if None in r_list:
                # print(r_list)
                none_idx = r_list.index(None)
                day_count = 0
                for i in range(none_idx, len(r_list)):
                    day_count += 1
                    if isinstance(r_list[i], torch.Tensor):
                        profit = r_list[i]
                        r_list[none_idx:i] = [(profit / day_count).float()
                                              for _ in range(none_idx, i)]
            else:
                idx = first_none_idx
                for trans in self.memory[first_none_idx:]:
                    trans.reward = r_list[idx]
                    idx += 1
                break

    def normalize_reward(self):
        r_list = np.array([trans.reward.item() for trans in self.memory])
        r_list = normalize(r_list.reshape(1, -1), norm='max').squeeze()

        # Create a copy of the memory so we don't modify the
        # actual memory when normalizing reward.
        memory_copy = copy.deepcopy(self.memory)

        for idx, trans in enumerate(memory_copy):
            trans.reward = torch.FloatTensor([r_list[idx]])

        return memory_copy

