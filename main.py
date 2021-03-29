import torch
import sacred

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data import load_env
from agent import Agent

invest_ex = Experiment('interp')


@invest_ex.config
def base_config():
    gamma = 0.99
    start_learning = 10
    memory_size = 1000000
    batch_size = 32
    target_update_step = 10
    test_interval = 100

    epsilon = 1
    epsilon_minimum = 0.05
    epsilon_decay_rate = 0.9999
    learning_rate = 0.001

    # Maximum update steps
    max_step = 40000000

    n_actions = 2



@invest_ex.main
def run(gamma, start_learning, memory_size, batch_size, target_update_step, test_interval,
        epsilon, epsilon_decay_rate, learning_rate, max_step, n_actions):
    env = load_env()
    agent = Agent(env, gamma, start_learning, memory_size, batch_size, target_update_step, test_interval,
                  epsilon, epsilon_decay_rate, learning_rate, max_step, n_actions)
    agent.learn()


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    invest_ex.observers.append(FileStorageObserver('runs'))
    invest_ex.run_commandline()
