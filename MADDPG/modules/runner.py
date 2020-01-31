import time
import imageio
import numpy as np
from utils.misc import time_left, make_folders
from wrappers.multiagent_wrapper import Multiagents


class Simple_Runner:
    def __init__(self, args, logger):
        self.args = args
        self.eval_stack = []
        self.logger = logger
        self.agents = Multiagents(args, logger)
        self.path = make_folders('./result', self.args)
        for k in sorted(vars(args).keys()):
            print("{}: {}".format(k, vars(args)[k]))

        self.has_adv, self.has_agent = False, False
        if 'adversary' in self.args.atypes:
            self.adv_idx = np.where(np.array(self.args.atypes) == 'adversary')[0][0]
            self.has_adv = True
        if 'agent' in self.args.atypes:
            self.agent_idx = np.where(np.array(self.args.atypes) == 'agent')[0][0]
            self.has_agent = True

    def train(self, env, eval_env):
        start_time = 0
        end_time = 0

        tot_rwds = []
        tot_steps = 0
        actor_losses = [0]
        critic_losses = [0]
        critic_values = [0]

        ep_num = 0
        ep_rwd = 0
        ep_step = 0

        self.eval(eval_env=eval_env, render_saving=False)
        self.agents.reset()
        obs = env.reset()

        for _ in range(self.args.max_steps):

            explr_pct_remaining = max(0, 25000 - ep_num) / 25000
            self.agents.rescale(0.0 + (0.3 - 0.0) * explr_pct_remaining)
            self.agents.reset()

            if tot_steps > self.args.start_timesteps:
                acts = self.agents.get_actions(obs, noise=True, batch=False)
            else:
                acts = env.sample_action()

            acts, rwd, nxt_obs, tm, info = env.step(acts)
            # assert np.array(obs).squeeze().shape == nxt_obs.shape

            # Learning Part
            if tot_steps > self.args.start_timesteps and (tot_steps % self.args.steps_per_update) < 1:
                end_time = time.time()
                al, cl, q = np.array([0.0]), np.array([0.0]), np.array([0.0])
                for _ in range(self.args.num_update):
                    loss = self.agents.learning()
                    if all(np.array(loss['flag'])):
                        al = al + np.array(loss['loss_p'])
                        cl = cl + np.array(loss['loss_q'])
                        q = q + np.array(loss['val_q'])
                        self.agents.update_target()

                if all(np.array(loss['flag'])):
                    if self.has_adv:
                        self.logger.log_scalar(f"Ep/ADV_critic", cl[self.adv_idx].item())
                        self.logger.log_scalar(f"Ep/ADV_policy", al[self.adv_idx].item())
                    if self.has_agent:
                        self.logger.log_scalar(f"Ep/AGENT_critic", cl[self.agent_idx].item())
                        self.logger.log_scalar(f"Ep/AGENT_policy", al[self.agent_idx].item())

                    actor_losses.append(np.mean(al).item())
                    critic_losses.append(np.mean(cl).item())
                    critic_values.append(np.mean(q).item())
                    start_time = time.time()

            ep_rwd += rwd
            ep_step += 1
            tot_steps += 1

            # Logging statistics
            if tm.any() or (self.args.finish_at_max and self.args.roll_out_steps == ep_step):
                tm = ~tm
                self.agents.add_to_buffer(np.array(obs).squeeze(), acts, rwd, nxt_obs, tm, ep_step)
                tot_rwds.append(ep_rwd)
                ep_num += 1
                duration = end_time - start_time

                """Cooperative Setting. BUT Competitive Setting?!"""
                if self.has_adv:
                    self.logger.log_scalar(f"Ep/ADV_rwd", ep_rwd[self.adv_idx])
                if self.has_agent:
                    self.logger.log_scalar(f"Ep/AGENT_rwd", ep_rwd[self.agent_idx])

                # self.logger.log_scalar("Tot/step", tot_steps)
                self.logger.log_scalar("Ep/step", ep_step)
                self.logger.log_scalar("Avg/rwd", np.mean(tot_rwds).item())
                self.logger.log_scalar("Avg/rwd_std", np.std(tot_rwds).item())
                self.logger.log_scalar("Ep/rwd", np.mean(ep_rwd).item())
                self.logger.log_scalar("Avg/critic_loss", np.mean(critic_losses).item())
                self.logger.log_scalar("Avg/policy_loss", np.mean(actor_losses).item())

                print(f"Ep: {ep_num} | Avg_rwd: {np.mean(tot_rwds).item():.2f} std: {np.std(tot_rwds).item():.3f}",
                      f"| Ep_rwd: {np.mean(ep_rwd).item():.2f} | Qval: {critic_values[-1]:.2f}",
                      f"| PL: {actor_losses[-1]:.2f} | CL: {critic_losses[-1]:.2f}",
                      f"Remained: {(self.args.max_steps - tot_steps) / self.args.max_steps * 100:.2f}%",
                      f"@{time_left(duration, self.args.max_steps, self.args.steps_per_update)}"
                      )

                ep_rwd, ep_step = 0, 0
                self.agents.reset()
                obs = env.reset()
            else:
                self.agents.add_to_buffer(np.array(obs).squeeze(), acts, rwd, nxt_obs, tm, ep_step)
                obs = nxt_obs

            # Periodically Evaluation
            if self.args.evaluation and (tot_steps + 1) % self.args.eval_req == 0:
                self.eval(eval_env=eval_env, render_saving=False)

        if self.args.model_saving:  # and self.best_eval < avg_rwd:
            self.agents.model_save(f'{self.path}/model')

    def eval(self, eval_env, render_saving):
        avg_rwd = 0.
        for n_ev in range(self.args.num_eval):
            obs, frames, steps = eval_env.reset(), [], 0
            while True:
                if render_saving:
                    frames.append(eval_env.render('rgb_array')[0])
                    eval_env.render()

                acts = self.agents.get_actions(obs, noise=False, batch=False)
                acts, rwd, nxt_obs, tm, info = eval_env.step(acts)
                avg_rwd += rwd
                obs = nxt_obs
                steps += 1

                if tm.any() or (self.args.finish_at_max and steps == self.args.eval_roll_out_steps):
                    break

            if render_saving:
                imageio.mimsave(f'{self.path}/gif/result_{n_ev}.gif', frames)
                print(f"result_{n_ev}.gif Saved @{self.path}/gif")
        avg_rwd /= self.args.num_eval

        """Cooperative Setting. BUT Cooperative Setting??"""
        print(avg_rwd)
        if self.has_adv:
            self.logger.log_scalar(f"Ep/ADV_eval_rwd", avg_rwd[self.adv_idx])
        if self.has_agent:
            self.logger.log_scalar(f"Ep/AGENT_eval_rwd", avg_rwd[self.agent_idx])

        avg_rwd = np.mean(avg_rwd).item()
        self.eval_stack.append(avg_rwd)

        print("-"*20)
        print("{:d}th Evaluation over {:d} episodes: {:.3f}".format(
            len(self.eval_stack), self.args.num_eval, avg_rwd))
        self.logger.log_scalar("Ep/eval_rwd", avg_rwd)
        self.logger.log_scalar("Avg/eval_rwd", np.mean(self.eval_stack).item())
        self.logger.log_scalar("Avg/eval_rwd_std", np.std(self.eval_stack).item())
        print("-"*20)

    def model_load(self):
        self.agents.model_load(f'{self.path}/model')
