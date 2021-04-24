import copy
import os
import collections
import time
import weakref
import os.path as osp
import numpy as np
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import skopt
import ray
from ray import tune
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.skopt import SkOptSearch

from main import invest_ex
from utils import sacred_copy, update


tune_ex = Experiment('tune', ingredients=[invest_ex])

output_root = 'runs/tune_runs'
cwd = os.getcwd()

@tune_ex.config
def base_config():
    exp_name = 'exp'
    exp_ident = 'transfer-amazn-v-jpm'
    metric = 'avg_reward'
    tune_mode = 'max'

    use_skopt = False
    skopt_search_mode = 'max'
    skopt_space = collections.OrderedDict([
        # Below are some examples of ways you can declare opt variables.
        #
        # Using a log-uniform prior between some bounds:
        # ('lr', (1e-4, 1e-2, 'log-uniform')),
        #
        # Using a uniform distribution over integers:
        # ('nrollouts', (30, 150)),
        #
        # Using just a single value:
        # ('batch_size', [32, 64, 128]),
        ('batch_size', [32, 64, 128]),
        ('learning_rate', (1e-4, 1e-2, 'log-uniform')),
        ('gamma', (0.99, 0.999, 'log-uniform')),
        ('memory_size', (335, 3350)),
        ('target_update_step', (10, 100)),
        ('epsilon_decay_rate', (0.999, 0.99999, 'log-uniform')),
        ('epsilon_minimum', (0.001, 0.1, 'log-uniform')),
        ('n_episodes', (200, 2000)),
    ])
    skopt_ref_configs = []

    spec = dict(
        policy_path=tune.grid_search(['/userhome/cs/cyn0531/COMP3359_proj/runs/invest_runs/1/policy_final.pth', None]),
        start_date=tune.grid_search(["2015-11-10"])
    )

    tune_run_kwargs = dict(
        num_samples=3,
        resources_per_trial=dict(
            cpu=1,
            gpu=0.02,
        )
    )


def run_exp(config, log_dir, exp_ident):
    # copy config so that we don't mutate in-place
    config = copy.deepcopy(config)
    config['root_dir'] = cwd
    config['exp_ident'] = exp_ident
    # if config['ticker'] == 'jpm':
    #     config['start_date'] = '1999-11-10'
    #     print('Modified start_date as', config['start_date'])
    from main import invest_ex

    observer = FileStorageObserver(os.path.join(cwd, log_dir))
    invest_ex.observers.append(observer)
    ret_val = invest_ex.run(config_updates=config)
    report_experiment_result(ret_val.result)
    return ret_val.result


def report_experiment_result(sacred_result):
    """To be run after an experiment."""
    filtered_result = {
        k: v
        for k, v in sacred_result.items() if isinstance(v, (int, float))
    }
    print(
        f"Got sacred result with keys {', '.join(filtered_result.keys())}")
    tune.report(**filtered_result)


class CheckpointFIFOScheduler(FIFOScheduler):
    """Variant of FIFOScheduler that periodically saves the given search
    algorithm. Useful for, e.g., SkOptSearch, where it is helpful to be able to
    re-instantiate the search object later on."""

    def __init__(self, search_alg):
        self.search_alg = weakref.proxy(search_alg)

    def on_trial_complete(self, trial_runner, trial, result):
        rv = super().on_trial_complete(trial_runner, trial, result)
        # references to _local_checkpoint_dir and _session_dir are a bit hacky
        checkpoint_path = os.path.join(
            trial_runner._local_checkpoint_dir,
            f'search-alg-{trial_runner._session_str}.pkl')
        self.search_alg.save(checkpoint_path + '.tmp')
        os.rename(checkpoint_path + '.tmp', checkpoint_path)
        # breakpoint()
        print("ckpt path", checkpoint_path)
        return rv


@tune_ex.main
def run(exp_ident, exp_name, metric, tune_mode, spec, tune_run_kwargs, use_skopt, skopt_search_mode,
        skopt_space, skopt_ref_configs):
    spec = sacred_copy(spec)
    log_dir = tune_ex.observers[0].dir

    ray.init()

    def trainable_function(config):
        # "config" is passed in by Ray Tune
        run_exp(config, log_dir, exp_ident)

    if use_skopt:
        assert skopt_search_mode in {'min', 'max'}, \
            'skopt_search_mode must be "min" or "max", as appropriate for ' \
            'the metric being optimised'
        assert len(skopt_space) > 0, "was passed an empty skopt_space"

        # do some sacred_copy() calls to ensure that we don't accidentally put
        # a ReadOnlyDict or ReadOnlyList into our optimizer
        skopt_space = sacred_copy(skopt_space)
        skopt_search_mode = sacred_copy(skopt_search_mode)
        skopt_ref_configs = sacred_copy(skopt_ref_configs)
        metric = sacred_copy(metric)

        # In addition to the actual spaces we're searching over, we also need
        # to store the baseline config values in Ray to avoid Ray issue #12048.
        # base_space = skopt.space.Categorical([base_config])
        # skopt_space['base_config'] = base_space
        # for ref_config in skopt_ref_configs:
        #     ref_config['+base_config'] = base_config

        sorted_space = collections.OrderedDict([
            (key, value) for key, value in sorted(skopt_space.items())
        ])
        for k, v in list(sorted_space.items()):
            # Cast each value in sorted_space to a skopt Dimension object, then
            # make the name of the Dimension object match the corresponding
            # key. This is the step that converts tuple ranges---like `(1e-3,
            # 2.0, 'log-uniform')`---to actual skopt `Space` objects.
            try:
                new_v = skopt.space.check_dimension(v)
            except ValueError:
                # Raise actually-informative value error instead
                raise ValueError(f"Dimension issue: k:{k} v: {v}")
            new_v.name = k
            sorted_space[k] = new_v

        skopt_optimiser = skopt.optimizer.Optimizer([*sorted_space.values()],
                                                    base_estimator='RF')
        algo = SkOptSearch(skopt_optimiser,
                           list(sorted_space.keys()),
                           metric=metric,
                           mode=skopt_search_mode,
                           points_to_evaluate=[[
                               ref_config_dict[k] for k in sorted_space.keys()
                           ] for ref_config_dict in skopt_ref_configs])
        tune_run_kwargs = {
            'search_alg': algo,
            'scheduler': CheckpointFIFOScheduler(algo),
            **tune_run_kwargs,
        }
        # completely remove 'spec'
        if spec:
            print("Will ignore everything in 'spec' argument")
        spec = {}

    tune_run = tune.run(
        trainable_function,
        name=exp_name,
        config=spec,
        local_dir=tune_ex.observers[0].dir,
        **tune_run_kwargs,
    )

    best_config = tune_run.get_best_config(metric=metric, mode=tune_mode)
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
