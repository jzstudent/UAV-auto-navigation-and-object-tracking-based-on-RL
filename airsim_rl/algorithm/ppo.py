import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def huber_loss(e, d):
    a = (abs(e)<=d).float()
    b = (e>d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def get_p_and_g_mean_norm(it):
    size = 1e-8
    su_p = 0
    su_g = 0
    for x in it:
        if x.grad is None:continue
        size += 1.
        su_p += x.norm()
        su_g += x.grad.norm()
    return su_p / size, su_g / size

class PPO():
    def __init__(self,                 
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 data_chunk_length,
                 value_loss_coef,
                 entropy_coef,
                 logger = None,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_huber_loss=True,
                 huber_delta=10.0,
                 device=None):

        self.step = 0
        self.logger = logger
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.data_chunk_length = data_chunk_length

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)#,weight_decay=1e-5)

        self.device = device

        self.use_huber_loss=use_huber_loss
        self.huber_delta=huber_delta
        self.crit=torch.nn.SmoothL1Loss()

    def update(self, rollouts):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        #advantages = (advantages - advantages.mean()) / (
            #    advantages.std() + 1e-5)

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch,  inform_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                if len(self.actor_critic.obs_shape)==1:
                    values, action_log_probs, dist_entropy, _= \
                    self.actor_critic.evaluate_actions( obs_batch,
                                                        recurrent_hidden_states_batch, masks_batch,actions_batch)
                elif len(self.actor_critic.obs_shape)==3:
                    values, action_log_probs, dist_entropy, _ = \
                        self.actor_critic.evaluate_actions([obs_batch,inform_batch],
                                                           recurrent_hidden_states_batch,
                                                           masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()


                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        value_pred_clipped = value_preds_batch \
                                             + (values - value_preds_batch).clamp(
                                             -self.clip_param, self.clip_param)

                        value_losses_clipped = self.crit(return_batch, value_pred_clipped)

                        value_losses = self.crit(return_batch, values)

                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_pred_clipped = value_preds_batch \
                                             + (values - value_preds_batch).clamp(
                                             -self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                else:
                    if self.use_huber_loss:
                        value_loss=self.crit(values,return_batch)
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                (value_loss * self.value_loss_coef+action_loss - dist_entropy * self.entropy_coef).backward()
                norm, grad_norm = get_p_and_g_mean_norm(self.actor_critic.parameters())
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)
                '''
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-self.max_grad_norm,self.max_grad_norm)
                '''
                self.optimizer.step()



        if self.logger is not None:

            self.logger.add_scalars('value_loss',
                                    {'value_loss': value_loss},
                                    self.step)
            self.logger.add_scalars('action_loss',
                                    {'action_loss': action_loss},
                                    self.step)
            self.logger.add_scalars('dist_entropy',
                                    {'dist_entropy': dist_entropy},
                                    self.step)
            self.logger.add_scalars('grad_norm',
                                    {'grad_norm': grad_norm},
                                    self.step)


        self.step += 1

