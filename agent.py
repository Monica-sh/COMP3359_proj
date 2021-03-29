import math
import random
from time import time

import torch
import torch.nn.functional as F

from network import MLPPolicy
from memory import Transition, ReplayMemory
from utils import AverageMeter


class Agent:
    def __init__(self,
                 env,
                 logger,
                 gamma,
                 start_learning,
                 memory_size,
                 batch_size,
                 target_update_step,
                 test_interval,
                 init_epsilon,
                 epsilon_decay_rate,
                 epsilon_decay_step,
                 learning_rate,
                 n_episodes,
                 n_actions):

        self.env = env
        self.gamma = gamma
        self.start_learning = start_learning
        self.batch_size = batch_size
        self.target_update_step = target_update_step
        self.test_interval = test_interval
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_step = epsilon_decay_step
        self.n_episodes = n_episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions

        self.policy_net = MLPPolicy(n_actions, env.state_shape).to(self.device)
        self.target_net = MLPPolicy(n_actions, env.state_shape).to(self.device)
        self.optimizer = torch.optim.Adam(lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.logger = logger
        self.epsilon = init_epsilon

    def experience_replay(self, DEBUG=False):
        """
        Input(s) :
        - policy_net: Policy DQN
        - target_net: Target DQN
        - memory: Transition memory buffer
        - optimizer: optimizer used by dqn model
        - params: dictionary storing global parameters.
                  expected parameter:
                  - params["BATCH_SIZE"]: input batch size
                  - params["GAMMA"]: discount rate in Q(s,a) = r + gamma * max_a'( Q(s',a') )
        - DEBUG: print debug messages if true
        Output(s) :
        - loss value compute from the sampled input transition batch.
        """

        # Skip training DQN model if there are not enough saved transitions in the memory buffer
        # to give a input batch.
        if len(self.memory) < self.batch_size:
            # Return a loss value = 0 to notice that training is not yet started (only for logging)
            return torch.tensor([0])

        ##### Prepare Transition Data Batch #####
        # Randomly sample BATCH_SIZE Transition tuples (state, action, reward, next_state),
        # each of (state, action, reward, next_state) is a PyTorch tensor.
        # Shapes of tensors (in each 4-tuple in "transitions"):
        #    state: (1, N_STATES), action: (1, 1), reward: (1), next_state: (1, N_STATES)
        # "transitions" is then a python list of 4-tuples with length BATCH_SIZE.
        transitions = self.memory.sample(self.batch_size)
        # Convert the python list of Transition tuples ("transitions") to one single
        # Transititon tuple (state, action, reward, next_state) ("batch").
        # Each of (state, action, reward, next_state) in "batch" is a list (with length BATCH_SIZE)
        # of tensors of that field in this data batch.
        batch = Transition(*zip(*transitions))

        # For each list of tensors (batch.[state/action/next_state/reward]) in "batch",
        # convert the list into one single tensor ([state/action/next_state/reward]_batch).
        #
        # torch.cat(...) is used to concatenate a list of B tensors with shape (1, N) to
        # a tensor with shape (B, N)
        state_batch = torch.cat(batch.state)  # shape: (B, N_STATES)
        action_batch = torch.cat(batch.action)  # shape: (B, 1)
        reward_batch = torch.cat(batch.reward)  # shape: (B)
        # Now, state_batch has the shape of input data to the DQN model.

        # However, it is not enough to simply concatenate batch.next_state (list of smaller tensors)
        # as next_state_batch (one larger tensor).
        # This is because if the terminal state is achieved, next_state is marked as None.
        # Also, later when we try to predict Q(s',a') for next state s', DQL is not designed to accept
        # terminal next state, and we will just set the term "gamma * max_a'( Q(s',a') )" to 0.
        #
        # Thus, we extract the non-final next state first, which will be used to predict Q(s',a') of
        # non-final next states later.
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])  # shape: (B_non_final)
        # To keep track of the non-final next states, we use a mask to mark down locations of values
        # of non-final next states.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        # shape: (B)
        if DEBUG:
            print("State batch: \n", state_batch)
            print("Action batch: \n", action_batch)
            print("Reward batch: \n", reward_batch)
            print("Locations of non-final next states: \n", non_final_mask)
            print("-----")

        ##### Deep Q Learning #####
        # Recall the equation of Q Learning:
        #     Q(s,a) = r + gamma * max_a'( Q(s',a') )

        ### LHS: Q(s,a) ###
        # policy_net(state_batch) :                                                                # shape: (B, N_ACTIONS)
        #     Policy DQN predicts Q values of current state Q(s,a) for all possible actions a.
        # policy_net(state_batch).gather(1, action_batch):
        #     As we are only interested in updating the Q value Q(s,a) of the taken action a,
        #     we gather the corresponding the concerned Q value according to index of action in action_batch.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # shape: (B,1)
        if DEBUG:
            print("Predicted Q values (LHS) = Q(s,a)")
            print("= ", state_action_values)

        ### RHS: r + gamma * max_a'( Q(s',a') ) ###
        # next_state_values :
        #     prepare a 0-value tensor for later to store the max predicted Q values max_a'(Q(s',a')).
        next_state_values = torch.zeros(self.batch_size, device=self.device)  # shape: (B)
        # target_net(non_final_next_states), shape (B_non_final) :
        #     Target DQN predicts Q values Q(s',a') for all possible actions a' (for non-final next state only)
        # target_net(non_final_next_states).max(1)[0].detach(), shape (B_non_final)
        #     max value of predicted Q values max_a'(Q(s',a'))
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() :
        #     update only values in next_state_values that correspond to non-final next states
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()  # shape: (B)
        # expected_state_action_values :
        #     target Q values = r + gamma * max_a'( Q(s',a') )
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)  # shape: (B)
        if DEBUG:
            print("Target Q values (RHS) = r + gamma * max_a'( Q(s',a') )")
            print("= ", expected_state_action_values)

        ##### Update Network Weights #####
        # Compute the loss between predicted Q values (LHS) and target Q values (RHS).
        # Mean Squared Error (MSE) is used as the loss function:
        #     loss = (LHS - RHS)^2
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

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
        if global_step <= self.epsilon_decay_step:
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
        if random.random() <= self.epsilon:
            # With prob. epsilon,
            # (Exploration) select random action.

            # Your task:
            # 1. Pick a random action
            # 2. Prepare the action as a tensor with type long and shape (1,1)
            # (Hint: you may consider random.randrange(...))

            action = random.randrange(0, self.n_actions, 1)
            action = torch.LongTensor([[action]]).to(self.device)

        else:
            # With prob. 1 - epsilon,
            # (Exploitation) select action with max predicted Q-Values of current state.

            # Your task:
            # 1. Predict Q values of current state
            # 2. Select action with greatest Q value
            # 3. Prepare the action as a tensor with type long and shape (1,1)
            # (Hint: policy_net(state) outputs the Q values for all actions)
            with torch.no_grad():
                action = torch.argmax(self.policy_net(state)).unsqueeze(0).unsqueeze(0).to(self.device)

        return action

    def train(self):
        self.policy_net.train()  # Set Policy DQN model as train mode
        start_time = time()  # Timer
        global_steps = 0
        for episode in range(self.n_episodes):
            if episode % 100 == 0:
                print("===== Episode {} =====".format(episode))
            ##### 2.1. (Game Starts) Initialization of Mountain Car Environment #####
            # Initialize the environment, get initial state
            state = self.env.reset()

            ##### 2.2. Loop for Steps #####
            # Logging for current episode
            done = None  # To mark if current episode is done
            episode_reward = 0  # Sum of rewards received in current episode
            loss_meter = AverageMeter()

            # Loop till end of episode (done = True)
            while not done:
                self.get_epsilon(global_steps)

                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action[0][0].item())

                if done:
                    next_state = None

                # reward: convert to tensor with shape (1)
                reward = torch.tensor([reward], device=self.device)

                self.memory.push(state, action, reward, next_state)

                loss = self.experience_replay(DEBUG=False)
                loss_meter.update(loss)

                if global_steps % self.target_update_step == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Update training results at the end of episode.
                state = next_state
                global_steps += 1
                episode_reward += reward.item()

            # Logging after an episode
            end_time = time()

            self.logger.record({'reward': episode_reward,
                                'loss': loss_meter.avg})

            # Print out logging messages
            if episode % 100 == 0:
                print("Time: ", end_time - start_time)
                print("Global Steps: ", global_steps)
                print("Epsilon: ", self.epsilon)
                print("Reward: ", episode_reward)
                print("====================")