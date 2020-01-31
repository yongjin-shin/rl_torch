import numpy as np
from collections import deque
from utils.misc import Singleton
import torch


class Buffer:
    def __init__(self, args):
        self.args = args
        self.buffer_size = self.args.buffer
        self.batch_size = self.args.batch
        self.buffer = deque()
        self.rwds_contianer = np.zeros(shape=(self.args.buffer, 1))
        self.rwd_mean = [0] * 1
        self.rwd_std = [0] * 1
        self.count = 0

    def add(self, args):
        singleton = Singleton(*args)
        if self.count >= self.buffer_size:
            self.buffer_pop()
            self.buffer_append(singleton)
        else:
            self.rwds_contianer[self.count] = args[-4]
            self.buffer_append(singleton)

    def get_samples(self):
        """
        Todo
            1) Different reward mean/ std
            since MPE is weird collision, it should be changed
            And also, for competitive setting, reward should be different.
        """
        assert self.can_samples()

        if self.args.norm_rwd:
            assert self.count <= self.buffer_size
            self.rwd_mean = np.mean(self.rwds_contianer[:self.count, :])
            self.rwd_std = np.std(self.rwds_contianer[:self.count, :])

        idx = np.random.choice(np.arange(len(self.buffer)), self.batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        return samples

    def can_samples(self):
        return self.count >= self.batch_size

    def buffer_pop(self):
        self.buffer.popleft()
        self.count -= 1
        assert self.count == len(self.buffer)

    def buffer_append(self, item):
        self.buffer.append(item)
        self.count += 1
        assert self.count == len(self.buffer)

    def __repr__(self):
        return "Buffer. Buffer Size:{} Batch Size:{} Count elements: {}".\
            format(self.buffer_size, self.batch_size, self.count)


class MYBuffer:
    def __init__(self, args):
        self.args = args
        self.buffer_size = self.args.buffer
        self.batch_size = self.args.batch
        self.buffer = deque()
        self.rwds_contianer = np.zeros(shape=(self.args.buffer, 1))
        self.rwd_mean = [0] * 1
        self.rwd_std = [0] * 1
        self.count = 0

    def add(self, args):
        singleton = Singleton(*args)
        if self.count >= self.buffer_size:
            self.buffer_pop()
            self.buffer_append(singleton)
        else:
            self.rwds_contianer[self.count] = args[-4][0][0]
            self.buffer_append(singleton)

    def get_samples(self, idx):
        assert self.can_samples()
        # idx = np.random.choice(np.arange(len(self.buffer)), self.batch_size, replace=False)

        if self.args.norm_rwd:
            assert self.count <= self.buffer_size
            self.rwd_mean = np.mean(self.rwds_contianer[:self.count, :])
            self.rwd_std = np.std(self.rwds_contianer[:self.count, :])

        samples = [self.buffer[i] for i in idx]
        s = Singleton(*zip(*samples))
        assert s is not None
        list_sample = [
            [torch.FloatTensor(np.array(s.obs).squeeze()[:, i]) for i in range(3)],
            [torch.FloatTensor(np.array(s.act).squeeze()[:, i]) for i in range(3)],
            [torch.FloatTensor((np.array(s.rwd).squeeze()[:, i] - self.rwd_mean) / self.rwd_std) for i in range(3)],
            [torch.FloatTensor(np.array(s.nxt_obs).squeeze()[:, i]) for i in range(3)],
            [torch.FloatTensor(np.array(s.tm).squeeze()[:, i]) for i in range(3)]
        ]
        return list_sample

    def can_samples(self):
        return self.count >= self.batch_size

    def buffer_pop(self):
        self.buffer.popleft()
        self.count -= 1
        assert self.count == len(self.buffer)

    def buffer_append(self, item):
        self.buffer.append(item)
        self.count += 1
        assert self.count == len(self.buffer)

    def get_average_rewards(self, N):
        if self.count == 1e6:
            inds = np.arange(self.count - N, self.count)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.count - N), self.count)
        return np.sum(np.array([self.buffer[i].rwd for i in inds]).squeeze(), axis=0)

    def __repr__(self):
        return "Buffer. Buffer Size:{} Batch Size:{} Count elements: {}".\
            format(self.buffer_size, self.batch_size, self.count)

    def __len__(self):
        return len(self.buffer)
