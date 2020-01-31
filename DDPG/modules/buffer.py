import random
import numpy as np
from collections import deque, namedtuple


Singleton = namedtuple('Singleton', ('obs', 'act', 'rwd', 'nxt_obs', 'tm', 't'))


class Buffer:
    def __init__(self, args):
        self.args = args
        self.buffer_size = self.args.buffer
        self.batch_size = self.args.batch
        self.buffer = deque()
        self.count = 0

    def add(self, args):
        singleton = Singleton(*args)
        if self.count >= self.buffer_size:
            self.buffer_pop()
            self.buffer_append(singleton)
        else:
            self.buffer_append(singleton)

    def get_samples(self):
        assert self.can_samples()
        return random.sample(self.buffer, self.batch_size)

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