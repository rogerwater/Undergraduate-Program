import numpy as np
import torch


class Preprocessor:
    def __init__(self, shape, device='cpu'):
        self.shape = (shape[-1], shape[0], shape[1])

        self.device = device
        self.rms = RunningMeanStd(shape=(1,) + self.shape)

    def __call__(self, x):
        x = np.asarray(x).reshape(x.shape[0], *self.shape)
        self.rms.update(x)
        x = x - self.rms.mean
        return torch.FloatTensor(x).to(self.device)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_count)

    def update_from_moments(self, batch_mean, batch_count):
        self.mean, self.count = update_mean_var_count_from_moments(
            self.mean, self.count, batch_mean, batch_count)


def update_mean_var_count_from_moments(mean, count, batch_mean, batch_count):
    delta = batch_mean - mean
    total_count = count + batch_count

    new_mean = mean + delta * batch_count / total_count
    return new_mean, total_count
