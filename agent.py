import math
import random
from time import time

import torch
import torch.nn.functional as F

from network import MLPPolicy


class Agent:
    def __init__(self,
                 env,
                 gamma,
                 start_learning,
                 memory_size,
                 batch_size,
                 target_update_step,
                 test_interval,
                 epsilon,
                 epsilon_decay_rate,
                 learning_rate,
                 max_step,
                 n_actions):

        self.env = env
        self.gamma = gamma
        self.start_learning = start_learning
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_step = target_update_step
        self.test_interval = test_interval
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.learning_rate = learning_rate
        self.max_step = max_step

        self.policy = MLPPolicy(n_actions, env.state_shape)
        self.optimizer = 0
        self.memory = 0

    def experience_replay(self, policy_net, target_net, memory, optimizer, params, DEBUG=False):
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
        if DEBUG:
            print("===== Start of Experience Replay =====")
        # Get global parameters
        BATCH_SIZE = params["BATCH_SIZE"]  # Input batch size
        GAMMA = params["GAMMA"]  # Discount rate in Q(s,a) = r + gamma * max_a'( Q(s',a') )

        # Skip training DQN model if there are not enough saved transitions in the memory buffer
        # to give a input batch.
        if len(memory) < BATCH_SIZE:
            # Return a loss value = 0 to notice that training is not yet started (only for logging)
            return torch.tensor([0])

        device = next(policy_net.parameters()).device  # Get computation device used by DQN model

        ##### Prepare Transition Data Batch #####
        # Randomly sample BATCH_SIZE Transition tuples (state, action, reward, next_state),
        # each of (state, action, reward, next_state) is a PyTorch tensor.
        # Shapes of tensors (in each 4-tuple in "transitions"):
        #    state: (1, N_STATES), action: (1, 1), reward: (1), next_state: (1, N_STATES)
        # "transitions" is then a python list of 4-tuples with length BATCH_SIZE.
        transitions = memory.sample(BATCH_SIZE)
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
                                                batch.next_state)), device=device, dtype=torch.bool)
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
        state_action_values = policy_net(state_batch).gather(1, action_batch)  # shape: (B,1)
        if DEBUG:
            print("Predicted Q values (LHS) = Q(s,a)")
            print("= ", state_action_values)

        ### RHS: r + gamma * max_a'( Q(s',a') ) ###
        # next_state_values :
        #     prepare a 0-value tensor for later to store the max predicted Q values max_a'(Q(s',a')).
        next_state_values = torch.zeros(BATCH_SIZE, device=device)  # shape: (B)
        # target_net(non_final_next_states), shape (B_non_final) :
        #     Target DQN predicts Q values Q(s',a') for all possible actions a' (for non-final next state only)
        # target_net(non_final_next_states).max(1)[0].detach(), shape (B_non_final)
        #     max value of predicted Q values max_a'(Q(s',a'))
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() :
        #     update only values in next_state_values that correspond to non-final next states
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()  # shape: (B)
        # expected_state_action_values :
        #     target Q values = r + gamma * max_a'( Q(s',a') )
        expected_state_action_values = reward_batch + (GAMMA * next_state_values)  # shape: (B)
        if DEBUG:
            print("Target Q values (RHS) = r + gamma * max_a'( Q(s',a') )")
            print("= ", expected_state_action_values)

        ##### Update Network Weights #####
        # Compute the loss between predicted Q values (LHS) and target Q values (RHS).
        # Mean Squared Error (MSE) is used as the loss function:
        #     loss = (LHS - RHS)^2
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Update of DQN network weights
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            # Gradients are clipped within range [-1,1], to prevent exploding magnitude of gradients
            # and failure of training.
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if DEBUG:
            print("Loss: ", loss)
            print("===== End of Experience Replay =====")
        # Return the computed loss value (for logging outside this function)
        return loss

    def get_epsilon(self, global_step, params):
        """
        Input(s) :
        - global_step: total number of steps taken so far from the beginning of training phase
        - params: dictionary of global parameters, expecting values:
                  - params["EPS_START"]: starting value of epsilon
                  - params["EPS_EN"]: min value of epsilon
                  - params["EPS_DECAY_STEPS"]: number of steps for epsilon to decay from EPS_START to EPS_END.
        Output(s):
        - epsilon value at global_step
        """
        EPS_START = params["EPS_START"]
        EPS_END = params["EPS_END"]
        EPS_DECAY_STEPS = params["EPS_DECAY_STEPS"]

        if global_step <= EPS_DECAY_STEPS:
            # When global_step <= EPS_DECAY_STEPS, epsilon is decaying linearly.
            return EPS_START - global_step * (EPS_START - EPS_END) / EPS_DECAY_STEPS
        else:
            # Otherwise, epsilon stops decaying and stay at its minimum value EPS_END
            return EPS_END

    """ Epsilon-Greedy Policy """

    def select_action(self, policy_net, state, epsilon, params):
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
        device = next(policy_net.parameters()).device  # Get computation device used by DQN model
        if random.random() <= epsilon:
            # With prob. epsilon,
            # (Exploration) select random action.

            # Your task:
            # 1. Pick a random action
            # 2. Prepare the action as a tensor with type long and shape (1,1)
            # (Hint: you may consider random.randrange(...))

            action = random.randrange(0, env.action_space.n, 1)
            action = torch.LongTensor([[action]]).to(device)

        else:
            # With prob. 1 - epsilon,
            # (Exploitation) select action with max predicted Q-Values of current state.

            # Your task:
            # 1. Predict Q values of current state
            # 2. Select action with greatest Q value
            # 3. Prepare the action as a tensor with type long and shape (1,1)
            # (Hint: policy_net(state) outputs the Q values for all actions)
            with torch.no_grad():
                action = torch.argmax(policy_net(state)).unsqueeze(0).unsqueeze(0).to(device)

        return action

    def train(self):
        policy_net.train()  # Set Policy DQN model as train mode
        start_time = time()  # Timer
        for episode in range(PARAMS["N_EPISODES"]):
            if episode % 100 == 0:
                print("===== Episode {} =====".format(episode))
            ##### 2.1. (Game Starts) Initialization of Mountain Car Environment #####
            # Initialize the environment, get initial state
            state = env.reset()
            # Preprocess state
            state = preprocess_state(state, device)

            ##### 2.2. Loop for Steps #####
            # Logging for current episode
            done = None  # To mark if current episode is done
            episode_steps = 0  # Counter of steps taken in current episode
            episode_reward = 0  # Sum of rewards received in current episode
            episode_max_x = -100  # Record the max x car position achieved in current episode

            # Loop till end of episode (done = True)
            while not done:
                ##### 2.2.1. (Epsilon-Greedy) Select Action #####
                # ------------------------- Sub-Task 1 -------------------------
                # Get epsilon value based on total number of steps taken during entire training phase
                # Returned epsilon is a float value.
                epsilon = get_epsilon(global_steps, PARAMS)

                # ------------------------- End of Sub-Task 1 -------------------------

                # ------------------------- Sub-Task 2 -------------------------
                # Select action with epsilon-greedy policy using Policy DQN.
                # Returned action is a tensor with shape (1,1).
                action = select_action(policy_net, state, epsilon, PARAMS)

                # ------------------------- End of Sub-Task 2 -------------------------

                ##### 2.2.2. Take Action #####
                # ------------------------- Sub-Task 3 -------------------------
                # Take action and get observations (next_state, rewards, done)
                next_state, reward, done, info = env.step(action[0][0].item())

                # ------------------------- End of Sub-Task 3 -------------------------
                # Adjust reward received to foster training of DQN model
                reward = adjust_reward(reward, next_state)

                ##### 2.2.3. (Experiment Replay) Store Transition #####
                # Before storing (state, action, next_state, rewards) to memory buffer,
                # convert the values into PyTorch tensors.
                # state: converted to tensor from previous iteration
                # action: prepared as tensor in function select_action(...)
                # next_state:
                if not done:
                    # If next state is not a terminal state, next_state will be memorized.
                    # Preprocess next_state before saving to memory.
                    next_state = preprocess_state(next_state, device)
                else:
                    # If next state is a terminal state, mark next_state as None.
                    # Later during Experience Replay, the corresponding Q values will be
                    # set to 0s.
                    next_state = None
                # reward: convert to tensor with shape (1)
                reward = torch.tensor([reward], device=device)

                # ------------------------- Sub-Task 4 -------------------------
                # Store the transition (s,a,r,s') in memory
                memory.push(state, action, reward, next_state)

                # ------------------------- End of Sub-Task 4 -------------------------

                ##### 2.2.4. (Experiment Replay) Train DQN Model by sampling a Transitions Batch from Memory #####
                # ------------------------- Sub-Task 5 -------------------------
                # Note that update network weights of Policy DQN occurs here.
                experience_replay(policy_net, target_net, memory, optimizer, PARAMS, DEBUG=False)

                # ------------------------- End of Sub-Task 5 -------------------------

                ##### 2.2.5 Update Target DQN network weights #####
                # Only update Target DQN once in every TARGET_UPDATE_PER_STEPS steps in the entire training phase
                if global_steps % PARAMS["TARGET_UPDATE_PER_STEPS"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                ##### 2.2.6 End of Epsiode #####
                # Update training results at the end of episode.
                state = next_state
                global_steps += 1
                episode_reward += reward.item()
                episode_steps += 1
                if next_state is not None and next_state[0, 0] > episode_max_x:
                    episode_max_x = next_state[0, 0].item()

                    # If too many steps are taken in this episode, forcibly stop this episode.
                # This it to avoid the current episode ends up looping.
                if episode_steps > PARAMS["MAX_STEP_PER_EPISODE"]:
                    # However, this is not triggered in this example, because
                    # because PARAMS["MAX_STEP_PER_EPISODE"] is set to be equal to
                    # the default max number of steps of Mountain Car environment.
                    # This checking is left here only as a remark of such scenario.
                    break

            # Logging after an episode
            end_time = time()
            all_rewards.append(episode_reward)
            all_steps.append(episode_steps)
            all_max_xs.append(episode_max_x)

            # Print out logging messages
            if episode % 100 == 0:
                print("Time: ", end_time - start_time)
                print("Steps: ", episode_steps)
                print("Global Steps: ", global_steps)
                print("Epsilon: ", epsilon)
                print("Reward: ", episode_reward)
                print("Max x Pos:", episode_max_x)
                print("====================")