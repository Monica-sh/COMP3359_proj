import json
import os
import copy
import collections
import pickle
import torch


class Logger:
    def __init__(self, save_root):
        self.save_root = save_root
        self.progress_file = os.path.join(self.save_root, 'progress.json')
        with open(self.progress_file, 'w') as outfile:
            json.dump({}, outfile)

    def record(self, dic):
        with open(self.progress_file) as json_file:
            data = json.load(json_file)

        for key, value in dic.items():
            if key in data.keys():
                data[key].append(value)
            else:
                data[key] = [value]

        with open(self.progress_file, 'w') as outfile:
            json.dump(data, outfile)

    def get_avg_reward(self):
        with open(self.progress_file) as json_file:
            data = json.load(json_file)

        return sum(data['reward'])/len(data['reward'])

    def save_model(self, model):
        torch.save(model, os.path.join(self.save_root, 'policy_final.pth'))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sacred_copy(o):
    """Perform a deep copy on nested dictionaries and lists.
    If `d` is an instance of dict or list, copies `d` to a dict or list
    where the values are recursively copied using `sacred_copy`. Otherwise, `d`
    is copied using `copy.deepcopy`. Note this intentionally loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.
    :param o: (object) if dict, copy recursively; otherwise, use `copy.deepcopy`.
    :return A deep copy of d."""
    if isinstance(o, dict):
        return {k: sacred_copy(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [sacred_copy(v) for v in o]
    else:
        return copy.deepcopy(o)


def update(d, *updates):
    """Recursive dictionary update (pure)."""
    d = copy.copy(d)  # to make this pure
    for u in updates:
        for k, v in u.items():
            if isinstance(d.get(k), collections.Mapping):
                # recursive insert into a mapping
                d[k] = update(d[k], v)
            else:
                # if the existing value is not a mapping, then overwrite it
                d[k] = v
    return d

