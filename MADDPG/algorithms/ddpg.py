from modules.networks import Actor, Critic, Update, Target
from modules.buffer import Buffer
from modules.learner import Learner
from utils.misc import Singleton, Normal, OUProcess, gumbel_softmax, onehot


class DDPG:
    def __init__(self, args, logger, action_space=None):
        self.args = args
        self.device = self.args.device
        self.logger = logger

        if self.args.noise == 'normal':
            self.noise = Normal(args=self.args)
        elif args.noise == 'ou':
            self.noise = OUProcess(args=args)
        else:
            raise NotImplementedError

        self.ac_update = Update(Actor(self.args).to(self.device), Critic(self.args).to(self.device))
        self.ac_target = Target(Actor(self.args).to(self.device), Critic(self.args).to(self.device))
        self.replay_buffer = Buffer(self.args)
        self.learner = Learner(self.args, self.ac_update)
        self.ac_target.eval()
        self.initialize()
        self.print_model()

    def get_all_networks(self, all_update, all_target, cnt):
        if self.args.alg_myself == 'MADDPG':
            self.learner.get_all_networks(all_update, all_target, cnt)
            # print(f'{self.args.alg_myself}/Agent ID {self.args.aid}: Updates & Targets are shared')

    def initialize(self):
        self.ac_target.load_state_dict(self.ac_update)
        self.ac_target.eval()
        self.noise.reset()
        print(f'{self.args.aid}th agent has been initialized: from DDPG.py')

    def reset(self):
        self.noise.reset()

    def get_actions(self, obs, noise=True, batch=False, hard=False):
        acts = self.ac_update.get_action(obs, batch)
        if self.args.discrete_action:
            if noise:
                assert batch is False
                acts = gumbel_softmax(acts, hard).cpu().detach().numpy().squeeze()
            else:
                acts = onehot(acts)
        else:
            acts = acts.cpu().detach().numpy().squeeze()
            if noise:
                assert batch is False
                acts = self.noise.get_action(acts)
        return acts

    def add_to_buffer(self, *args):
        self.replay_buffer.add(args)

    def learning(self):
        flag = False
        loss_p, loss_q, val_q = None, None, None
        if self.replay_buffer.can_samples():
            samples = self.replay_buffer.get_samples()
            batch = Singleton(*zip(*samples))

            # Update Critic
            critic_mse_loss, qval = self.learner.get_critic_loss(batch, self.ac_update, self.ac_target,
                                                                 self.replay_buffer.rwd_mean, self.replay_buffer.rwd_std)
            self.learner.optimize_critic(critic_mse_loss, self.ac_update)

            # Update Actor
            policy_loss = self.learner.get_policy_loss(batch, self.ac_update)
            self.learner.optimize_actor(policy_loss, self.ac_update)

            loss_p = policy_loss.item()
            loss_q = critic_mse_loss.item()
            val_q = qval.item()
            flag = True
        return {'loss_p': loss_p, 'loss_q': loss_q, 'flag': [flag], 'val_q': val_q}

    def update_target(self):
        # Update Target Networks
        self.learner.soft_update_params(update=self.ac_update, target=self.ac_target)

        # if self.args.bn:
        #     self.learner.batch_norm_update(update=self.ac_update, target=self.ac_target)

    def print_model(self):
        print("===== Update =====")
        print(self.ac_update.actor)
        print(self.ac_update.critic)
        print("===== Target =====")
        print(self.ac_target.actor)
        print(self.ac_target.critic)
        print("="*50)

    def model_save(self, path):
        self.ac_update.model_save(path, self.args.aid)
        self.ac_target.model_save(path, self.args.aid)

    def model_load(self, path):
        self.ac_update.model_load(path, self.args.aid)
        self.ac_target.model_load(path, self.args.aid)
