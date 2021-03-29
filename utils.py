import json
import os


class Logger:
    def __init__(self, save_root):
        self.save_root = save_root
        self.progress_file = os.path.join(self.save_root, 'progress.json')
        with open(self.progress_file, 'w') as outfile:
            json.dump({}, outfile)

    def record(self, dic):
        with open(self.progress_file) as json_file:
            data = json.load(json_file)

        for key, value in dic:
            if key in data.keys():
                data[key].append(value)
            else:
                data[key] = [value]

        with open(self.progress_file, 'w') as outfile:
            json.dump(data, outfile)


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
