import torch
import torch.optim as optim
import torch.nn as nn


class Learner:
    def __init__(self, args, update):
        self.args = args
        assert update.iam == 'update'
        # self.A_opt = optim.RMSprop(update.actor.parameters(),
        #                            lr=self.args.actor_lr)
        # self.C_opt = optim.RMSprop(update.critic.parameters(),
        #                            lr=self.args.critic_lr, weight_decay=self.args.critic_l2)
        self.A_opt = optim.Adam(update.actor.parameters(), lr=self.args.actor_lr)
        self.C_opt = optim.Adam(update.critic.parameters(), lr=self.args.critic_lr,
                                weight_decay=self.args.critic_l2)

    def get_critic_loss(self, batch, update, target):
        """
        - obs_batch : [N obs_dim]
        - acts_batch : [N act_dim]
        - q_val_batch : [N 1]
        - rwd_batch : [N 1]
        - nxt_obs_batch : [N obs_dim]
        - nxt_acts_batch : [N act_dim]
        - nxt_q_val : [N 1]
        - ys : [N 1]
        """
        update.critic.train()
        target.eval()

        # get Q_{s+1}|target
        rwd_batch = torch.tensor(list(batch.rwd), device=self.args.device)
        nxt_obs_batch = torch.tensor(list(batch.nxt_obs), device=self.args.device)
        nxt_acts_batch = target.get_action(nxt_obs_batch, batch=True)
        nxt_q_val = target.get_qval(nxt_obs_batch, nxt_acts_batch.detach())

        # get Q_{s}|update
        obs_batch = torch.tensor(list(batch.obs), device=self.args.device)
        acts_batch = torch.tensor(list(batch.act), device=self.args.device)
        q_val = update.get_qval(obs_batch, acts_batch)

        # Last state's Q_{s+1} should be zero
        tm_batch = torch.tensor(list(batch.tm), device=self.args.device)
        last_state_mask = ~tm_batch
        ys = rwd_batch + self.args.gamma * nxt_q_val * last_state_mask

        # we only need Q_update updated
        # thus, let ys be detached
        ys = ys.detach()

        # get mse: 1/N (Q_{s} - Q_{s+1})^2
        # assert torch.eq(ys.size(), q_val.size())
        return torch.mean((ys - q_val) ** 2), q_val

    def optimize_critic(self, loss, update):
        self.C_opt.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(update.critic.parameters(),
        #                          self.args.critic_gradient_clip)
        self.C_opt.step()

    def get_policy_loss(self, batch, update):
        update.actor.train()
        update.critic.eval()
        obs_batch = torch.tensor(list(batch.obs), device=self.args.device)

        acts_batch = update.get_action(obs_batch, batch=True)
        qval = update.get_qval(obs_batch, acts_batch)
        return torch.mean(-qval)

    def optimize_actor(self, loss, update):
        self.A_opt.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(update.actor.parameters(),
        #                          self.args.actor_gradient_clip)
        self.A_opt.step()

    def soft_update_params(self, update, target):
        for up_act_params, up_crt_params, tg_act_params, tg_crt_params in zip(update.actor.parameters(),
                                                                              update.critic.parameters(),
                                                                              target.actor.parameters(),
                                                                              target.critic.parameters()):
            tg_act_params.data.copy_(self.args.tau * up_act_params.data + (1.0 - self.args.tau) * tg_act_params.data)
            tg_crt_params.data.copy_(self.args.tau * up_crt_params.data + (1.0 - self.args.tau) * tg_crt_params.data)

    @staticmethod
    def batch_norm_update(update, target):
        target.actor.bn1.running_mean = torch.clone(update.actor.bn1.running_mean)
        target.actor.bn1.running_var = torch.clone(update.actor.bn1.running_var)
        target.actor.bn1.num_batches_tracked = torch.clone(update.actor.bn1.num_batches_tracked)
        target.actor.bn2.running_mean = torch.clone(update.actor.bn2.running_mean)
        target.actor.bn2.running_var = torch.clone(update.actor.bn2.running_var)
        target.actor.bn2.num_batches_tracked = torch.clone(update.actor.bn2.num_batches_tracked)

        target.critic.bn1.running_mean = torch.clone(update.critic.bn1.running_mean)
        target.critic.bn1.running_var = torch.clone(update.critic.bn1.running_var)
        target.critic.bn1.num_batches_tracked = torch.clone(update.critic.bn1.num_batches_tracked)
