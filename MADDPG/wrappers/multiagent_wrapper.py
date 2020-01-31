from algorithms import REGISTRY as algorithms
from utils.misc import Loss_n
from copy import deepcopy
import numpy as np


class Multiagents:
    def __init__(self, args, logger, action_space=None):
        self.multiargs = args
        self.n = self.multiargs.n_agents
        self.agents = select_algorithms(args, logger, action_space)
        self.num_update = 0

    def rescale(self, ss):
        for i in range(self.n):
            self.agents[i].noise.scale = ss

    def reset(self):
        for i in range(self.n):
            self.agents[i].reset()

    def get_actions(self, obs_n, noise, batch):
        if self.n < 2:
            return [self.agents[0].get_actions(obs_n, noise, batch)]
        else:
            action_n = []
            for i in range(self.n):
                assert self.agents[i].args.aid == i
                action_n.append(self.agents[i].get_actions(np.atleast_2d(obs_n[i]), noise, batch))
            return action_n

    # Singleton = namedtuple('Singleton', ('obs', 'act', 'rwd', 'nxt_obs', 'tm', 't'))
    # self.agents.add_to_buffer(np.array(obs).squeeze(), acts, rwd, nxt_obs, tm, ep_step)
    def add_to_buffer(self, *items):
        if self.n == 1:
            self.agents[0].add_to_buffer(*items)
        else:
            for i in range(self.n):
                assert self.agents[i].args.aid == i
                if 'MADDPG' == self.agents[i].args.alg_myself:
                    # obs = np.copy(items[0])
                    # act = np.copy(items[1])
                    # nxt_obs = np.copy(items[3])

                    obs = np.concatenate(items[0])
                    act = np.concatenate(items[1])
                    nxt_obs = np.concatenate(items[3])
                else:
                    obs = np.copy(items[0][i])
                    act = np.copy(items[1][i])
                    nxt_obs = np.copy(items[3][i])
                rwd = np.copy([items[2][i]])
                tm = np.copy([items[4][i]])
                t = np.copy(items[5])
                self.agents[i].add_to_buffer(obs, act, rwd, nxt_obs, tm, t)

    def learning(self):
        all_update = [self.agents[i].ac_update for i in range(self.n)]
        all_target = [self.agents[i].ac_target for i in range(self.n)]
        for j in range(self.n):
            self.agents[j].get_all_networks(all_update, all_target, self.num_update)

        loss = []
        for i in range(self.n):
            assert self.agents[i].args.aid == i

            self.agents[i].ac_update.train()
            self.agents[i].ac_target.train()

            ls = self.agents[i].learning()
            loss.append(Loss_n(ls['loss_p'], ls['loss_q'], ls['flag'], ls['val_q']))

            self.agents[i].ac_target.eval()
            self.agents[i].ac_update.eval()

        loss = Loss_n(*zip(*loss))
        self.num_update += 1

        return loss._asdict()

    def update_target(self):
        for i in range(self.n):
            self.agents[i].update_target()

    def model_save(self, path):
        for i in range(self.n):
            self.agents[i].model_save(path)

    def model_load(self, path):
        for i in range(self.n):
            self.agents[i].model_load(path)
            print(f'{path} was loaded')


def agent_identification(multiargs, agent_id, alg):
    args = deepcopy(multiargs)
    args.aid = agent_id
    if args.scenario_env is 'openai':
        args.n_p_obs = args.n_obs[agent_id]
        args.n_p_actions = args.n_actions[agent_id]
        if not args.discrete_action:
            args.action_high = args.action_high[agent_id]
            args.action_low = args.action_low[agent_id]

        if alg == 'MADDPG':
            args.n_q_obs = sum(multiargs.n_obs)
            args.n_q_actions = sum(multiargs.n_actions)
            args.alg_myself = 'MADDPG'
        elif alg == 'DDPG':
            args.n_q_obs = args.n_p_obs
            args.n_q_actions = args.n_p_actions
            args.alg_myself = 'DDPG'
        else:
            raise NotImplementedError

        args.action_start_idx = [0] + list(np.cumsum(args.n_actions)[:-1])
        args.obs_start_idx = [0] + list(np.cumsum(args.n_obs)[:-1])
        args.n_obs = None
        args.n_actions = None
        args.atypes = args.atypes[args.aid]
    return args


def select_algorithms(args, logger, action_space):
    agents = []
    for aid, atype in enumerate(args.atypes):
        if atype == 'adversary':
            print(f"\n{atype.upper()}: {args.alg_adversary}")
            alg = args.alg_adversary
            agent = algorithms[alg](args=agent_identification(args, aid, alg), logger=logger, action_space=action_space)
        else:
            alg = args.alg_agent
            print(f"\n{atype.upper()}: {args.alg_agent}")
            agent = algorithms[args.alg_agent](args=agent_identification(args, aid, alg), logger=logger, action_space=action_space)
        agents.append(agent)
    return agents
