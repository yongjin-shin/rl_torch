import torch.nn as nn
import torch.nn.functional as F
import torch


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.device = self.args.device
        assert all(self.args.action_high + self.args.action_low == 0)
        self.action_bound = self.args.action_high

        # Torch does fan_in initializing for Linear
        self.fc1 = nn.Linear(self.args.n_obs, 400)
        self.fc2 = nn.Linear(400, 300)
        self.last = nn.Linear(300, self.args.n_actions)
        nn.init.uniform_(self.last.weight, -3 * 1e-3, 3 * 1e-3)
        nn.init.uniform_(self.last.bias, -3 * 1e-3, 3 * 1e-3)

        if args.bn:
            self.bn1 = nn.BatchNorm1d(400)
            self.bn2 = nn.BatchNorm1d(300)

    def forward(self, obs):
        if self.args.bn:
            h1 = F.relu(self.bn1(self.fc1(obs)))
            h2 = F.relu(self.bn2(self.fc2(h1)))
        else:
            h1 = F.relu(self.fc1(obs))
            h2 = F.relu(self.fc2(h1))
        out = torch.tanh(self.last(h2))
        scaled_out = out * torch.tensor(self.action_bound, device=self.device)
        return scaled_out


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.device = self.args.device

        self.fc1_o = nn.Linear(self.args.n_obs, 400)
        self.fc2_o = nn.Linear(400, 300, bias=False)
        self.fc2_a = nn.Linear(self.args.n_actions, 300, bias=False)
        self.bias = nn.Parameter(torch.ones([300]), requires_grad=True)
        self.last = nn.Linear(300, 1)
        nn.init.uniform_(self.last.weight, -3 * 1e-3, 3 * 1e-3)
        nn.init.uniform_(self.last.bias, -3 * 1e-3, 3 * 1e-3)

        if args.bn:
            self.bn1 = nn.BatchNorm1d(400)

    def forward(self, obs, act):
        if self.args.bn:
            h1 = F.relu(self.bn1(self.fc1_o(obs)))
        else:
            h1 = F.relu(self.fc1_o(obs))
        h2_o = self.fc2_o(h1)
        h2_a = self.fc2_a(act)
        h2 = F.relu(h2_o + h2_a + self.bias)
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
            obs = torch.tensor([obs])
        else:
            self.actor.train()
        return self.actor(obs.to(self.device).float())

    def get_qval(self, obs, acts):
        return self.critic(obs.to(self.device).float(),
                           acts.to(self.device).float())

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        assert self.iam == 'update'
        self.actor.train()
        self.critic.train()

    def load_state_dict(self, update):
        raise NotImplementedError()

    def state_dict(self):
        return self.actor.state_dict(), self.critic.state_dict()


class Target(Update):
    def __init__(self, actor, critic):
        super().__init__(actor, critic)
        self.iam = 'target'

    def load_state_dict(self, update):
        assert update.iam == 'update'
        a, c = update.state_dict()
        self.actor.load_state_dict(a)
        self.critic.load_state_dict(c)

    def get_action(self, obs, batch=False):
        if not batch:
            self.actor.eval()
            obs = torch.tensor([obs])
        return self.actor(obs.to(self.device).float())