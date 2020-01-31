from modules.networks import Actor, Critic, Update, Target
from modules.buffer import Buffer, Singleton
from modules.learner import Learner
from modules.utils import OUProcess, Normal


class DDPG:
    def __init__(self, args, logger, action_space):
        self.args = args
        self.device = args.device
        self.logger = logger

        if args.noise == 'ou':
            self.noise = OUProcess(action_space=action_space, args=args)
        elif args.noise == 'normal':
            self.noise = Normal(action_space=action_space, args=args)

        self.ac_update = Update(Actor(args).to(self.device), Critic(args).to(self.device))
        self.ac_target = Target(Actor(args).to(self.device), Critic(args).to(self.device))
        self.replay_buffer = Buffer(args)
        self.learner = Learner(args, self.ac_update)
        self.ac_target.load_state_dict(self.ac_update)
        self.ac_target.eval()

    def initialize(self):
        self.ac_target.load_state_dict(self.ac_update)
        self.ac_target.eval()
        self.noise.reset()
        print("Agent has been initialized.")

    def reset(self):
        self.noise.reset()

    def get_actions(self, obs, noise=True, batch=False):
        acts = self.ac_update.get_action(obs, batch)
        if noise:
            assert batch is False
            acts = self.noise.get_action(acts)
        return acts

    def add_to_buffer(self, *args):
        self.replay_buffer.add(args)

    def learning(self):
        flag = False
        loss_p, loss_q = None, None
        if self.replay_buffer.can_samples():
            samples = self.replay_buffer.get_samples()
            batch = Singleton(*zip(*samples))

            # Update Critic
            critic_mse_loss, qval = self.learner.get_critic_loss(batch, self.ac_update, self.ac_target)
            self.learner.optimize_critic(critic_mse_loss, self.ac_update)

            # Update Actor
            policy_loss = self.learner.get_policy_loss(batch, self.ac_update)
            self.learner.optimize_actor(policy_loss, self.ac_update)

            # ep_qval += qval.max().item()
            loss_p = policy_loss.item()
            loss_q = critic_mse_loss.item()
            flag = True
        return {'loss_p': loss_p, 'loss_q': loss_q, 'flag': flag}

    def update_target(self):
        # Update Target Networks
        self.learner.soft_update_params(update=self.ac_update, target=self.ac_target)

        if self.args.bn:
            self.learner.batch_norm_update(update=self.ac_update, target=self.ac_target)

    def print_model(self):
        print("===== Update =====")
        print(self.ac_update.actor)
        print(self.ac_update.critic)
        print("===== Target =====")
        print(self.ac_target.actor)
        print(self.ac_target.critic)
