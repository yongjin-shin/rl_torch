import gym
import numpy as np
from copy import deepcopy
from utils.make_env import make_env
from utils.misc import onehot_idx


mujoco = ['Ant-v2', 'Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Reacher-v2', 'InvertedPendulum-v2']
openai = ['simple', 'simple_adversary', 'simple_speaker_listener', 'simple_spread', 'simple_tag']


class Envrionment:
    def __init__(self, args, discrete_action, benchmark=False):
        self.args = args
        self.scenario_name = args.scenario
        if self.scenario_name in mujoco:
            args.scenario_env = 'mujoco'
            args.finish_at_max = False
            self.env = gym.make(self.scenario_name)
        else:
            args.scenario_env = 'openai'
            args.finish_at_max = True
            self.env = make_env(self.scenario_name, benchmark, discrete_action)

            if self.args.discrete_action:
                self.action_dims = [aspace.n for aspace in self.env.action_space]

            if all([hasattr(a, 'adversary') for a in self.env.agents]):
                self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                    self.env.agents]
            else:
                self.agent_types = ['agent' for _ in self.env.agents]

    def args_update(self):
        """
        Todo
            1) I still don't know how to deal w/ Box in gym
            Need to change n_obs/n_actions
        """
        if self.args.scenario_env is 'mujoco':
            self.args.n_agents = 1
            self.args.n_obs = np.prod(self.env.observation_space.shape)
            self.args.n_actions = np.prod(self.env.action_space.shape)
            self.args.action_high = self.env.action_space.high
            self.args.action_low = self.env.action_space.low
        else:
            self.args.n_agents = self.env.n
            self.args.n_obs = [np.prod(self.env.observation_space[i].shape) for i in range(self.args.n_agents)]
            if not self.args.discrete_action:
                self.args.n_actions = [np.prod(self.env.action_space[i].shape) for i in range(self.args.n_agents)]
                self.args.action_high = [self.env.action_space[i].high for i in range(self.args.n_agents)]
                self.args.action_low = [self.env.action_space[i].low for i in range(self.args.n_agents)]
            else:
                self.args.n_actions = [self.env.action_space[i].n for i in range(self.args.n_agents)]

            self.args.atypes = self.get_agent_types()
        return self.args

    def seed(self, seed_num):
        if self.args.scenario_env == 'openai':
            self.env._seed(seed_num)
        else:
            self.env.seed(seed_num)

    def reset(self):
        if self.args.scenario_env == 'openai':
            return np.array(self.env._reset()).squeeze()
        else:
            return self.env.reset()

    def sample_action(self):
        if self.args.scenario_env == 'openai':
            acts = []
            for i in range(self.args.n_agents):
                act = self.env.action_space[i].sample()
                if self.args.discrete_action:
                    acts.append(onehot_idx(act, self.action_dims[i]))
                else:
                    acts.append(act)
            return acts
        else:
            return self.env.action_space.sample()

    def step(self, acts):
        if self.args.scenario_env == 'openai':
            nxt_obs, rwd, tm, info = self.env.step(deepcopy(acts))
            return self.data_shaper(acts, nxt_obs, rwd, tm, info)
        else:
            nxt_obs, rwd, tm, info = self.env.step(acts)
            return self.data_shaper(acts, nxt_obs, rwd, tm, info)

    def data_shaper(self, actions, nxt_obs, rwd, tm, info):
        actions = np.atleast_1d(np.array(actions).squeeze())
        rwd = np.atleast_1d(np.array(rwd).squeeze())
        nxt_obs = np.atleast_1d(np.array(nxt_obs).squeeze())
        tm = np.atleast_1d(np.array(tm).squeeze())
        return actions, rwd, nxt_obs, tm, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def get_agent_types(self):
        assert self.args.scenario_env == 'openai'
        if all([hasattr(agent, 'adversary') for agent in self.env.agents]):
            atypes = ['adversary' if agent.adversary else 'agent' for agent in self.env.agents]
        else:
            atypes = ['agent' for _ in self.env.agents]

        return atypes



