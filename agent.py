import math
import pdb
import random
from time import time

import numpy as np
import torch
import torch.nn.functional as F

from network import MLPPolicy
from memory import ReplayMemory
from utils import AverageMeter
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self,
                 env,
                 logger,
                 gamma,
                 start_learning,
                 memory_size,
                 batch_size,
                 target_update_step,
                 policy_update_step,
                 max_episode_step,
                 init_epsilon,
                 epsilon_minimum,
                 epsilon_decay_rate,
                 epsilon_decay_step,
                 learning_rate,
                 n_episodes,
                 n_actions,
                 hidden_dim,
                 print_interval):

        self.env = env
        self.gamma = gamma
        self.start_learning = start_learning
        self.batch_size = batch_size
        self.target_update_step = target_update_step
        self.policy_update_step = policy_update_step
        self.max_episode_step = max_episode_step
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_step = epsilon_decay_step
        self.n_episodes = n_episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.print_interval = print_interval

        self.policy_net = MLPPolicy(hidden_dim, n_actions, env.state_shape).to(self.device).float().to(device)
        self.target_net = MLPPolicy(hidden_dim, n_actions, env.state_shape).to(self.device).float().to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size, env.state_shape)
        self.logger = logger
        self.epsilon = init_epsilon
        self.epsilon_minimum = epsilon_minimum

        self.memory_cache = ReplayMemory(self.max_episode_step, env.state_shape)

    def experience_replay(self, DEBUG=False):
        # Skip training DQN model if there are not enough saved transitions in the memory buffer
        # to give a input batch.
        if len(self.memory) < self.batch_size:
            # Return a loss value = 0 to notice that training is not yet started (only for logging)
            return torch.FloatTensor([0])

        # state batch shape: (B, N_STATES)
        # action batch shape: (B, 1)
        # reward batch shape: (B)
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(self.batch_size)

        # shape: (B)
        if DEBUG:
            print("State batch: \n", state_batch, "type: ", state_batch.type())  # # torch.FloatTensor
            print("Action batch: \n", action_batch, "type: ", action_batch.type())  # torch.LongTensor
            print("Reward batch: \n", reward_batch, "type: ", reward_batch.type())  # torch.FloatTensor
            print("-----")

        state_action_values = self.policy_net(state_batch).gather(1, action_batch).view(self.batch_size)
        if DEBUG:
            print("Predicted Q values (LHS) = Q(s,a)")
            print("= ", state_action_values)
            print("type: ", state_action_values.type())  # torch.FloatTensor

        # RHS: r + gamma * max_a'( Q(s',a') )
        next_state_values = []
        for next_state in next_state_batch:
            if next_state is not None:
                next_state_values.append(self.policy_net(next_state))
            else:
                next_state_values.append(torch.FloatTensor([0]))
        next_state_values = torch.max(torch.stack(next_state_values), dim=1)

        # expected_state_action_values :
        #     target Q values = r + gamma * max_a'( Q(s',a') )
        expected_state_action_values = (reward_batch + (self.gamma * next_state_values.values)).view(self.batch_size)
        if DEBUG:
            print("Target Q values (RHS) = r + gamma * max_a'( Q(s',a') )")
            print("= ", expected_state_action_values)
            print("type: ", expected_state_action_values.type())  # torch.FloatTensor

        # Update
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Update of DQN network weights
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # Gradients are clipped within range [-1,1], to prevent exploding magnitude of gradients
            # and failure of training.
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if DEBUG:
            print("Loss: ", loss)
            print("===== End of Experience Replay =====")
        # Return the computed loss value (for logging outside this function)
        return loss

    def get_epsilon(self, global_step):
        if global_step <= self.epsilon_decay_step and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay_rate

    def select_action(self, state):
        """
        Input(s) :
        - policy_net: Policy DQN for predicting Q values (for Exploitation)
        - state: current state for predicting Q values (for Exploitation)
        - epsilon: exploration probability
        - params: dictionary of global parameters, expecting:
                  - params["N_ACTIONS"]: number of possible actions
        Output(s) :
        - action: action to be taken, a tensor with type long and shape (1,1)
        """
        while True:
            if random.random() <= self.epsilon:
                # With prob. epsilon
                action = random.randrange(0, self.n_actions, 1)
                action = torch.LongTensor([[action]]).to(self.device)

            else:
                # With prob. 1 - epsilon,
                # (Exploitation) select action with max predicted Q-Values of current state.

                with torch.no_grad():
                    action = torch.argmax(self.policy_net(state)).unsqueeze(0).unsqueeze(0).to(self.device)

            # The agent can only sell stocks when it is holding some;
            # Similarly, it can only buy stocks when it's holding nothing
            # action = 2 >> buy, action = 1 >> no sell no buy, action = 0 >> sell
            # Only valid actions can be returned.
            if self.env.holding_stocks and action in [0, 1]:
                break
            elif not self.env.holding_stocks and action in [1, 2]:
                break

        return action

    def train(self):
        self.policy_net.train()  # Set Policy DQN model as train mode
        start_time = time()  # Timer
        global_steps = 0
        for episode in range(self.n_episodes):
            # Initialize the environment, get initial state
            # you can change the beginning date here
            state = self.env.reset(date="2016-11-10")
            # preprocess state
            state = preprocess_state(state, self.device)

            # Logging for current episode
            done = None  # To mark if current episode is done
            episode_reward = 0  # Sum of rewards received in current episode
            episode_step = 0  # Cumulative steps in current episode
            loss_meter = AverageMeter()

            # Loop till end of episode (done = True or when step reaches max)
            while not done and episode_step < self.max_episode_step:
                self.get_epsilon(global_steps)

                action = self.select_action(state)

                next_state, reward, done = self.env.step(action[0][0].item())

                if not done:
                    # preprocess next_state
                    next_state = preprocess_state(next_state, self.device)
                else:
                    next_state = [None]

                self.memory_cache.push(state, action, [reward], next_state)

                if reward is not None:
                    self.memory_cache.process_reward()
                    push_length = self.memory_cache.position
                    self.memory.push(self.memory_cache.state[:push_length],
                                     self.memory_cache.action[:push_length],
                                     self.memory_cache.reward[:push_length],
                                     self.memory_cache.next_state[:push_length])
                    self.memory_cache.reset()

                    loss = self.experience_replay(DEBUG=False)

                    # print(f"Episode [{episode}/{self.n_episodes}] "
                    #       f"Global steps: {global_steps}, "
                    #       f"Episode steps: {episode_step}, "
                    #       f"loss: {loss}, "
                    #       f"Time elapsed: {str(datetime.timedelta(seconds=time() - start_time))}")
                    loss_meter.update(loss.item())

                if global_steps % self.target_update_step == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Update training results at the end of episode.
                state = next_state
                global_steps += 1
                episode_step += 1
                if reward:
                    episode_reward += reward

            # Logging after an episode
            end_time = time()

            self.logger.record({'reward': episode_reward,
                                'loss': loss_meter.avg})

            # Print out logging messages
            if episode % self.print_interval == 0:
                print("====================")
                print(f"Episode {episode}")
                print("Time: ", end_time - start_time)
                print("Global Steps: ", global_steps)
                print("Epsilon: ", self.epsilon)
                print("Loss: ", loss_meter.avg)
                print("Reward: ", episode_reward)
                print("====================")

        avg_reward = self.logger.get_avg_reward()
        self.logger.save_model(self.policy_net)
        return avg_reward
    

def preprocess_state(state, device=None):
    """
    To convert the state prepared by Gym and to a format
    that is convenient for later processing 
    (see comments in function Experience replay)

    Input(s) :
    - state: state numpy array prepared by Gym envrionment
    - device: computation device used for PyTorch tensors
    
    Output(s) :
    - state: state as a PyTorch tensor with type float and
             shape (1,2)
    """
    #                                           # The following values are default values.
    #                                           # variable type | value type | data shape
    # input state                               # numpy.ndarray |   float64  | (2,)  #! pandas series not numpy
    state = torch.from_numpy(np.array(state))   # torch.Tensor  |   double   | (2)  #! float64
    state = state.float()                       # torch.Tensor  |   float    | (2)  
    state = state.unsqueeze(0)                  # torch.Tensor  |   float    | (1,2)  

    # Pass state tensor to the specified computation device 
    # (if None, the default device is used)
    state = state.to(device)

    return state
