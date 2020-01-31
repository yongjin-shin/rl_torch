import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class Learner:
    def __init__(self, args, update):
        self.args = args
        assert update.iam == 'update'
        self.all_update = None
        self.all_target = None
        self.cnt = 0

        self.A_opt = optim.Adam(update.actor.parameters(), lr=self.args.actor_lr)

        if self.args.use_l2:
            self.C_opt = optim.Adam(update.critic.parameters(), lr=self.args.critic_lr,
                                    weight_decay=self.args.critic_l2)
        else:
            self.C_opt = optim.Adam(update.critic.parameters(), lr=self.args.critic_lr)

    def get_critic_loss(self, batch, update, target, rwd_mean, rwd_std):
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
        update.train()
        target.train()

        # get Q_{s+1}|target
        rwd_batch = torch.FloatTensor(np.array(batch.rwd)).to(self.args.device)

        if self.args.norm_rwd:
            assert rwd_std != 0
            rwd_batch = (rwd_batch - rwd_mean) / (rwd_std + 1e-10)

        nxt_obs_batch = torch.FloatTensor(np.array(batch.nxt_obs)).to(self.args.device)
        if self.args.alg_myself == 'MADDPG':
            all_nxt_acts_batch = []
            for single_id, other_target in enumerate(self.all_target):
                assert single_id == other_target.actor.args.aid
                s_idx = other_target.actor.args.obs_start_idx[single_id]
                e_idx = s_idx + other_target.actor.args.n_p_obs
                single_nxt_obs_batch = nxt_obs_batch[:, s_idx:e_idx]
                # single_nxt_obs_batch = nxt_obs_batch[:, single_id]
                if self.args.aid == single_id:
                    single_nxt_acts_batch = target.get_action(single_nxt_obs_batch, batch=True)
                    all_nxt_acts_batch.append(single_nxt_acts_batch)
                else:
                    single_nxt_acts_batch = other_target.get_action(single_nxt_obs_batch, batch=True)
                    all_nxt_acts_batch.append(single_nxt_acts_batch)
            nxt_acts_batch = torch.cat(all_nxt_acts_batch, dim=1)
        else:
            nxt_acts_batch = target.get_action(nxt_obs_batch, batch=True)
        nxt_q_val = target.get_qval(nxt_obs_batch.reshape((self.args.batch, -1)), nxt_acts_batch)

        # Last state's Q_{s+1} should be zero
        tm_batch = torch.tensor(np.array(batch.tm)).to(self.args.device)
        last_state_mask = ~tm_batch
        ys = rwd_batch + self.args.gamma * nxt_q_val * last_state_mask

        # we only need Q_update updated
        # thus, let ys be detached
        ys = ys.detach()

        # get Q_{s}|update
        obs_batch = torch.FloatTensor(np.array(batch.obs)).to(self.args.device)
        acts_batch = torch.FloatTensor(np.array(batch.act)).to(self.args.device)
        q_val = update.get_qval(obs_batch.reshape((self.args.batch, -1)),
                                acts_batch.reshape((self.args.batch, -1)))

        # get mse: 1/N (Q_{s} - Q_{s+1})^2
        return torch.mean((ys - q_val) ** 2), torch.mean(q_val)

    def optimize_critic(self, loss, update):
        self.C_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(update.critic.parameters(),
                                 self.args.critic_gradient_clip)
        self.C_opt.step()

    def get_policy_loss(self, batch, update):
        update.train()

        reg = 0
        obs_batch = torch.FloatTensor(np.array(batch.obs)).to(self.args.device)
        if self.args.alg_myself == 'MADDPG':
            all_acts_batch = []
            for single_id, other_update in enumerate(self.all_update):
                assert single_id == other_update.critic.args.aid
                s_idx = other_update.critic.args.obs_start_idx[single_id]
                e_idx = s_idx + other_update.critic.args.n_p_obs
                single_obs_batch = obs_batch[:, s_idx:e_idx]
                # single_obs_batch = obs_batch[:, single_id]
                if self.args.aid == single_id:
                    single_acts_batch = update.get_action(single_obs_batch, batch=True)
                    reg = torch.mean(single_acts_batch ** 2)
                    all_acts_batch.append(single_acts_batch)
                else:
                    single_acts_batch = other_update.get_action(single_obs_batch, batch=True)
                    all_acts_batch.append(single_acts_batch.detach())
            acts_batch = torch.cat(all_acts_batch, dim=1)
        else:
            acts_batch = update.get_action(obs_batch, batch=True)
            reg = torch.mean(acts_batch ** 2)

        qval = update.get_qval(obs_batch.reshape((self.args.batch, -1)),
                               acts_batch.reshape((self.args.batch, -1)))
        ret = torch.mean(-qval) + 1e-3 * reg
        return ret

    def optimize_actor(self, loss, update):
        self.A_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(update.actor.parameters(),
                                 self.args.actor_gradient_clip)
        self.A_opt.step()

    def soft_update_params(self, update, target):
        for up_act_params, up_crt_params, tg_act_params, tg_crt_params in zip(update.actor.parameters(),
                                                                              update.critic.parameters(),
                                                                              target.actor.parameters(),
                                                                              target.critic.parameters()):
            tg_act_params.data.copy_(self.args.tau * up_act_params.data + (1.0 - self.args.tau) * tg_act_params.data)
            tg_crt_params.data.copy_(self.args.tau * up_crt_params.data + (1.0 - self.args.tau) * tg_crt_params.data)

    def get_all_networks(self, all_update, all_target, cnt):
        if self.args.alg_myself == 'MADDPG':
            self.all_update = all_update
            self.all_target = all_target
            self.cnt = cnt
            assert self.all_update is not None and self.all_target is not None

            """For debugging"""
            # for i in range(len(self.all_update)):
            #     print(f'After sharing: {i}th agent critic last weight')
            #     print(self.all_target[i].critic.last.weight)

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
