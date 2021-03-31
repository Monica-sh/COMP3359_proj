import copy
import os
import time
import os.path as osp
import numpy as np
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import ray
from ray import tune

from main import invest_ex
from utils import sacred_copy, update


tune_ex = Experiment('tune', ingredients=[invest_ex])

output_root = 'runs/tune_runs'
cwd = os.getcwd()

@tune_ex.config
def base_config():
    exp_name = 'tune'
    metric = 'avg_reward'
    spec = dict(
        batch_size=tune.grid_search([32, 64, 128])
    )
    tune_run_kwargs = dict()


def run_exp(config, log_dir):
    # copy config so that we don't mutate in-place
    config = copy.deepcopy(config)
    config['root_dir'] = cwd
    from main import invest_ex

    observer = FileStorageObserver(osp.join(log_dir, 'tune'))
    invest_ex.observers.append(observer)
    ret_val = invest_ex.run(config_updates=config)
    return ret_val.result

@tune_ex.main
def run(exp_name, metric, spec, tune_run_kwargs):
    spec = sacred_copy(spec)
    log_dir = tune_ex.observers[0].dir

    ray.init()

    def trainable_function(config):
        # "config" is passed in by Ray Tune
        run_exp(config, log_dir)

    tune_run = tune.run(
        trainable_function,
        name=exp_name,
        config=spec,
        **tune_run_kwargs,
    )

    best_config = tune_run.get_best_config(metric=metric)
    print(f"Best config is: {best_config}")
    print("Results available at: ")
    print(tune_run._get_trial_paths())


def main():
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    observer = FileStorageObserver('runs/tune_runs')
    tune_ex.observers.append(observer)
    tune_ex.run_commandline()


if __name__ == '__main__':
    main()
