import torch
import sacred

from sacred import Experiment
from sacred.observers import FileStorageObserver

invest_ex = Experiment('interp')


@invest_ex.config
def base_config():
    gamma = 0.99
    start_learning = 10
    memory_size = 1000000
    batch_size = 32
    plot_every = 50
    reset_step = 10

    epsilon = 1
    epsilon_minimum = 0.05
    epsilon_decay_rate = 0.9999
    learning_rate = 0.001

    max_step = 40000000  # 40M steps max


@invest_ex.main
def run(gamma, start_learning, memory_size, batch_size, plot_every, reset_step, epsilon,
        epsilon_decay_rate, learning_rate, max_step):
    print("Do something here")


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    invest_ex.observers.append(FileStorageObserver('runs'))
    invest_ex.run_commandline()
