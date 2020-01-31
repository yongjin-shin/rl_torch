import torch.nn as nn
import torch.nn.functional as F
import torch


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.device = self.args.device
        if not self.args.discrete_action:
            assert all(self.args.action_high + self.args.action_low == 0)
            self.action_bound = torch.FloatTensor(self.args.action_high).to(self.device)

        if self.args.norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(self.args.n_p_obs)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        # Torch does fan_in initializing for Linear
        self.fc1 = nn.Linear(self.args.n_p_obs, self.args.hidden1)
        self.fc2 = nn.Linear(self.args.hidden1, self.args.hidden2)
        self.last = nn.Linear(self.args.hidden2, self.args.n_p_actions)

        if not self.args.discrete_action:
            nn.init.uniform_(self.last.weight, -3 * 1e-3, 3 * 1e-3)
            # nn.init.uniform_(self.last.bias, -3 * 1e-3, 3 * 1e-3)
            self.last_nonlin = lambda x: torch.tanh(x) * self.action_bound
        else:
            self.last_nonlin = lambda x: x

    def forward(self, obs):
        obs = self.in_fn(obs)
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        out = self.last_nonlin(self.last(h2))
        return out


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.device = self.args.device

        if self.args.norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(self.args.n_q_obs+self.args.n_q_actions)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(self.args.n_q_obs+self.args.n_q_actions, self.args.hidden1)
        self.fc2 = nn.Linear(self.args.hidden1, self.args.hidden2)
        self.last = nn.Linear(self.args.hidden2, 1)
        nn.init.uniform_(self.last.weight, -3 * 1e-3, 3 * 1e-3)
        nn.init.uniform_(self.last.bias, -3 * 1e-3, 3 * 1e-3)

    def forward(self, obs, act):
        q_in = torch.cat((obs, act), dim=1)
        q_in = self.in_fn(q_in)
        h1 = F.relu(self.fc1(q_in))
        h2 = F.relu(self.fc2(h1))
        ret = self.last(h2)
        return ret


class Update:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.device = actor.device
        self.iam = 'update'

    def get_action(self, obs, batch=False):
        if not batch:
            self.actor.eval()
            obs = torch.FloatTensor(obs).to(self.device)
        else:
            self.actor.train()
        return self.actor(obs)

    def get_qval(self, obs, acts):
        return self.critic(obs, acts)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def load_state_dict(self, update):
        raise NotImplementedError()

    def state_dict(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def model_save(self, path, aid):
        torch.save(self.actor.state_dict(), f'{path}/actor_{aid}.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic_{aid}.pth')

    def model_load(self, path, aid):
        self.actor.load_state_dict(torch.load(f'{path}/actor_{aid}.pth'))
        self.critic.load_state_dict(torch.load(f'{path}/critic_{aid}.pth'))


class Target(Update):
    def __init__(self, actor, critic):
        super().__init__(actor, critic)
        self.iam = 'target'

    def load_state_dict(self, update):
        assert update.iam == 'update'
        a, c = update.state_dict()
        self.actor.load_state_dict(a)
        self.critic.load_state_dict(c)
