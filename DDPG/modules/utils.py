import numpy as np
from time import ctime, time


class OUProcess:
    def __init__(self, action_space, args):
        """
        Todo
            1) I don't want to use 'action_space'
            what if Environment doesn't have action_space?
        """
        self.mu = args.mu
        self.theta = args.theta
        self.sigma = args.sigma
        self.action_space = action_space
        self.state = (np.ones(np.prod(self.action_space.shape)) * self.mu).reshape(self.action_space.shape)
        self.reset()

    def reset(self):
        self.state = (np.ones(np.prod(self.action_space.shape)) * self.mu).reshape(self.action_space.shape)

    def evolv_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action(self, action):
        assert len(action) == 1
        ou_state = self.evolv_state()
        return np.clip(action.detach().cpu().numpy()+ou_state, self.action_space.low, self.action_space.high)


class Normal(OUProcess):
    def __init__(self, action_space, args):
        super().__init__(action_space, args)
        self.args = args

    def evolv_state(self):
        return np.random.normal(0, self.args.action_high * self.args.expl_noise, size=self.args.n_actions)

    def reset(self):
        pass


def args_update(args, env):
    """
    Todo
        1) I still don't know how to deal w/ Box in gym
        Need to change n_obs/n_actions
    """
    args.n_obs = np.prod(env.observation_space.shape)
    args.n_actions = np.prod(env.action_space.shape)
    args.action_high = env.action_space.high
    args.action_low = env.action_space.low
    return args


def time_left(crnt_time, duration, remained, ep_step):
    exp_ep = int(remained/ep_step)
    exp_duration = exp_ep * duration
    return ctime(time() + exp_duration)
