import time
import numpy as np
from algorithms.ddpg import DDPG
from modules.utils import time_left


class Simple_Runner:
    def __init__(self, args, train_env, eval_env, logger):
        self.args = args
        self.env = train_env
        self.eval_env = eval_env
        self.eval_env.seed(self.args.seed + 100)

        self.eval_stack = []
        self.logger = logger
        self.agent = DDPG(args, logger, train_env.action_space)
        self.agent.initialize()
        self.agent.print_model()

    def train(self):
        tot_rwds = []
        tot_steps = 0
        max_steps = self.args.epoch * self.args.epoch_cycles * self.args.roll_out_steps

        # epoch_actions = []
        # epoch_qs = []
        actor_losses = []
        critic_losses = []

        ep_num = 0
        ep_rwd = 0
        ep_step = 0

        start_time = time.time()
        self.eval()
        self.agent.reset()
        obs = self.env.reset()
        for epoch in range(self.args.epoch):
            for t_cy in range(self.args.epoch_cycles):
                for t_roll in range(self.args.roll_out_steps):

                    if tot_steps > self.args.start_timesteps:
                        acts = self.agent.get_actions(obs, noise=True, batch=False)
                    else:
                        acts = self.env.action_space.sample()
                    nxt_obs, rwd, tm, info = self.env.step(acts)
                    obs, acts, rwd, nxt_obs, tm = self.need_moving_to_env_wrapper(obs, acts, rwd, nxt_obs, tm)
                    self.agent.add_to_buffer(obs, acts, rwd, nxt_obs, tm, t_roll)

                    ep_rwd += rwd[0]
                    ep_step += 1
                    tot_steps += 1
                    obs = nxt_obs

                    # Learning Part
                    loss = self.agent.learning()
                    if tot_steps > self.args.start_timesteps and loss['flag']:
                        actor_losses.append(loss['loss_p'])
                        critic_losses.append(loss['loss_q'])
                        self.agent.update_target()

                    if tm:
                        ep_num += 1
                        tot_rwds.append(ep_rwd)

                        end_time = time.time()
                        duration = end_time - start_time
                        self.logger.log_scalar("Tot/step", tot_steps)
                        self.logger.log_scalar("Avg/rwd", np.mean(tot_rwds).item())
                        self.logger.log_scalar("Avg/rwd_std", np.std(tot_rwds).item())

                        self.logger.log_scalar("Ep/step", ep_step)
                        self.logger.log_scalar("Ep/rwd", ep_rwd)

                        print("Ep: {:d} | Avg_rwd: {:.2f} std: {:.3f} | Ep_rwd: {:.2f} | Remained: {:d} | Duration: {:.2f} | Finshed@ {}".format(
                            ep_num, np.mean(tot_rwds).item(), np.std(tot_rwds).item(),
                            ep_rwd, max_steps - tot_steps, float(duration), time_left(end_time, duration, max_steps - tot_steps, ep_step)
                        ))

                        ep_rwd, ep_step = 0, 0
                        self.agent.reset()
                        obs = self.env.reset()
                        start_time = time.time()

                    if self.args.evaluation and (tot_steps+1) % self.args.eval_req == 0:
                        self.eval()

    def eval(self):
        avg_rwd = 0.
        for _ in range(self.args.num_eval):
            obs, tm = self.eval_env.reset(), False
            while not tm:
                acts = self.agent.get_actions(obs, noise=False, batch=False)
                nxt_obs, rwd, tm, info = self.eval_env.step(acts.detach().cpu().numpy().squeeze())
                avg_rwd += rwd
                obs = nxt_obs

        avg_rwd /= self.args.num_eval
        print("-"*20)
        print("Evaluation over {:d} episodes: {:.3f}".format(self.args.num_eval, avg_rwd))
        print("-"*20)
        self.eval_stack.append(avg_rwd)
        self.logger.log_scalar("Ep/eval_rwd", avg_rwd)
        self.logger.log_scalar("Avg/eval_rwd", np.mean(self.eval_stack).item())
        self.logger.log_scalar("Avg/eval_rwd_std", np.std(self.eval_stack).item())

    @staticmethod
    def need_moving_to_env_wrapper(obs, acts, rwd, nxt_obs, tm):
        """
        Todo
            Need to move this toward env wrapper
        """
        nxt_obs = nxt_obs.squeeze()
        rwd = np.atleast_1d(rwd.squeeze())
        acts = np.atleast_1d(acts.squeeze())
        tm = np.atleast_1d(tm)
        assert obs.shape == nxt_obs.shape
        return obs, acts, rwd, nxt_obs, tm

    # def reset(self):
    #     ep_rwd, ep_step_count = 0, 0
    #     actor_loss, critic_loss = [], []
    #
    #     self.agent.reset()
    #     obs = self.env.reset()
    #     return ep_rwd, ep_step_count, actor_loss, critic_loss, obs

# from utils import OUProcess
# from networks import Actor, Critic, Target, Update
# from buffer import Buffer, Singleton
# from learner import Learner
# class Runner:
#     def __init__(self, args, env, logger):
#         self.args = args
#         self.env = env
#         self.logger = logger
#         self.ou = OUProcess(action_space=self.env.action_space,
#                             mu=self.args.mu, theta=self.args.theta, sigma=self.args.sigma)
#         self.replay_buffer = Buffer(args)
#         self.update = Update(Actor(args), Critic(args))
#         self.target = Target(Actor(args), Critic(args))
#         self.learner = Learner(args, self.update)
#         self.target.load_state_dict(self.update)
#         self.target.eval()
#
#     def run(self):
#         """
#         Todo
#             Need to do something on Evaluation.
#             How can I calculate proper performance of evaluation?
#         """
#         for t_it in range(self.args.train_iter):
#             self.train()
#
#             if self.args.evaluation:
#                 for e_it in range(self.args.eval_iter):
#                     print("")
#                     # self.eval()
#
#     def reset(self):
#         ep_rwd = 0
#         ep_step_count = 0
#         loss_p = []
#         loss_q = []
#
#         self.ou.reset()
#         obs = self.env.reset()
#         return ep_rwd, ep_step_count, loss_p, loss_q, obs
#
#     def train(self):
#         train_sum_rwd, train_step_count, train_num_episode = 0, 0, 0
#
#         while train_step_count <= self.args.train_step_max:
#             ep_rwd, ep_step_count, loss_p, loss_q, obs = self.reset()
#
#             for t_el in range(self.args.ep_step_len):
#                 acts = self.update.get_action(obs)
#                 acts = self.ou.get_action(acts)
#                 nxt_obs, rwd, tm, info = self.env.step(acts)
#
#                 nxt_obs = nxt_obs.squeeze()
#                 rwd = np.atleast_1d(rwd.squeeze())
#                 acts = np.atleast_1d(acts.squeeze())
#                 tm = np.atleast_1d(tm)
#                 # assert obs.shape == nxt_obs.shape
#                 self.replay_buffer.add((obs, acts, rwd, nxt_obs, tm, t_el))
#
#                 if self.replay_buffer.can_samples():
#                     samples = self.replay_buffer.get_samples()
#                     batch = Singleton(*zip(*samples))
#
#                     # Update Critic
#                     critic_mse_loss, qval = self.learner.get_critic_loss(batch, self.update, self.target)
#                     self.learner.optimize_critic(critic_mse_loss, self.update)
#
#                     # Update Actor
#                     policy_loss = self.learner.get_policy_loss(batch, self.update)
#                     self.learner.optimize_actor(policy_loss, self.update)
#
#                     # Update Target Networks
#                     self.learner.soft_update_params(update=self.update, target=self.target)
#
#                     if self.args.bn:
#                         self.learner.batch_norm_update(update=self.update, target=self.target)
#
#                     # ep_qval += qval.max().item()
#                     loss_p.append(policy_loss.item())
#                     loss_q.append(critic_mse_loss.item())
#
#                 obs = nxt_obs
#                 ep_rwd += rwd
#                 ep_step_count += 1
#
#                 if tm:
#                     break
#
#             # update statistics
#             train_sum_rwd += ep_rwd
#             train_step_count += ep_step_count
#             train_num_episode += 1
#             train_avg_rwd = train_sum_rwd / train_num_episode if train_num_episode > 0 else 0
#             train_avg_stp = train_step_count / train_num_episode if train_num_episode > 0 else 0
#
#             # Log
#             self.logger.log_scalar("Train/policy_loss", np.mean(loss_p).item())
#             self.logger.log_scalar("Train/critic_loss", np.mean(loss_q).item())
#             self.logger.log_scalar("Train/episode_rwd", ep_rwd.item())
#             self.logger.log_scalar("Train/episode_len", ep_step_count)
#             self.logger.log_scalar("Train/Avg_rwd", train_avg_rwd.item())
#             self.logger.log_scalar("Train/Avg_stp", train_avg_stp)
#             print('Ep: {:d} | EpRwd: {:.3f} | AvgRwd: {:.3f} | EpSt: {:d} | AvgSt: {:.3f} | Remained: {:d}'.format(
#                 int(train_num_episode), ep_rwd.item(), train_avg_rwd.item(), ep_step_count, train_avg_stp,
#                 self.args.train_step_max - train_step_count))
