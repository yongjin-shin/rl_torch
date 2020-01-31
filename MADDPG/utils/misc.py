import os
import torch
from torch.distributions.uniform import Uniform
import numpy as np
from time import ctime, time
from collections import namedtuple

Singleton = namedtuple('Singleton', ('obs', 'act', 'rwd', 'nxt_obs', 'tm', 't'))
Loss_n = namedtuple('Loss_n', ('loss_p', 'loss_q', 'flag', 'val_q'))


class OUProcess:
    def __init__(self, args):
        """
        Todo
            1) I don't want to use 'action_space'
            what if Environment doesn't have action_space?
        """
        self.args = args
        self.mu = args.mu
        self.theta = args.theta
        self.sigma = args.sigma
        self.scale = self.args.scale
        self.action_dim = args.n_p_actions
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolv_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

    def get_action(self, action):
        ou_state = self.evolv_state()
        return np.clip(action + ou_state, self.args.action_low, self.args.action_high)


class Normal:
    def __init__(self, args):
        self.args = args

    def get_action(self, action):
        # assert len(action) == 1
        ou_state = self.evolv_state()
        return np.clip(action + ou_state, self.args.action_low, self.args.action_high)

    def evolv_state(self):
        return np.random.normal(0, self.args.action_high * self.args.expl_noise, size=self.args.n_p_actions)

    def reset(self):
        pass


def onehot_idx(idx, shape):
    ret = np.zeros(shape=shape)
    ret[idx] = 1
    return ret


def onehot(_input):
    ret = np.atleast_2d(np.zeros_like(_input.detach().cpu().numpy()))
    idx = _input.argmax(dim=-1)
    ret[:, idx] = 1
    return ret.squeeze()


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    unif = Uniform(0, 1)
    samples = unif.sample(shape)
    return -torch.log(-torch.log(samples + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits.to('cpu') + sample_gumbel(logits.size())
    return torch.softmax(y/temperature, dim=-1)


def gumbel_softmax(logits, hard, temperature=1.0):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y = onehot(y)
    return y


def time_left(duration, remained, ep_step):
    exp_ep = int(remained / ep_step)
    exp_duration = exp_ep * duration
    return ctime(time() + exp_duration)


def make_folders(path, args):
    if not os.path.exists(path):
        os.mkdir(path)

    path = f'{path}/{args.scenario}'
    if not os.path.exists(path):
        os.mkdir(f'{path}')
        print(f'{path} folder has been made.')

    import datetime
    now = datetime.datetime.now()
    path = f'{path}/{now:%Y-%m-%d-%H:%M}'
    if not os.path.exists(path):
        os.mkdir(f'{path}')
        print(f'{path} folder has been made.')

    if args.render_saving:
        if not os.path.exists(f'{path}/gif'):
            os.mkdir(f'{path}/gif')
            print(f'{path}/gif folder has been made.')

    if args.model_saving:
        if not os.path.exists(f'{path}/model'):
            os.mkdir(f'{path}/model')
            print(f'{path}/model folder has been made.')

    return path


def cuda_check(args):
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(int(torch.get_num_threads()))
    return device
