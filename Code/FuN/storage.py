import torch


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []

        self.size = size
        self.keys = keys
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for k in self.keys:
            setattr(self, k, [])

    def normalize(self, keys):
        for key in keys:
            k = torch.stack(getattr(self, key))
            k = (k - k.mean()) / (k.std() + 1e-10)
            setattr(self, key, [i for i in k])

    def stack(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.stack(x, dim=0), data)
