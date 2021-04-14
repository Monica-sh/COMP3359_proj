import os
import torch
import sacred

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data import load_env
from agent import Agent
from utils import Logger

invest_ex = Experiment('invest')


@invest_ex.config
def base_config():
    exp_ident = None
    gamma = 0.99
    start_learning = 10
    memory_size = 1800
    batch_size = 32
    target_update_step = 20
    policy_update_step = 3
    max_episode_step = 365

    init_epsilon = 1
    epsilon_minimum = 0.077
    epsilon_decay_rate = 0.9994
    epsilon_decay_step = 99999999999
    learning_rate = 0.0001

    n_episodes = 1000
    n_actions = 3
    hidden_dim = 24

    norm_state = True
    print_interval = 1

    root_dir = os.getcwd()


@invest_ex.main
def run(gamma, start_learning, memory_size, batch_size, target_update_step, policy_update_step,
        max_episode_step, init_epsilon, epsilon_minimum, epsilon_decay_rate, epsilon_decay_step,
        learning_rate, n_episodes, n_actions, norm_state, root_dir, hidden_dim, print_interval):

    env = load_env(root_dir, norm_state=norm_state)
    logger = Logger(invest_ex.observers[0].dir)
    agent = Agent(env, logger, gamma, start_learning, memory_size, batch_size, target_update_step,
                  policy_update_step, max_episode_step, init_epsilon, epsilon_minimum, epsilon_decay_rate,
                  epsilon_decay_step, learning_rate, n_episodes, n_actions, hidden_dim, print_interval)
    avg_reward = agent.train()
    return {'avg_reward': avg_reward}


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    invest_ex.observers.append(FileStorageObserver('runs/invest_runs'))
    invest_ex.run_commandline()
