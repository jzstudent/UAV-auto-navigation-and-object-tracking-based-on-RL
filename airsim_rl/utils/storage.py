import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, episode_length, n_rollout_threads, obs_shape, action_space,
                 recurrent_hidden_state_size):

        self.obs = torch.zeros(episode_length + 1, n_rollout_threads, *obs_shape)
        self.inform = torch.zeros(episode_length + 1, n_rollout_threads, 9)
        self.recurrent_hidden_states = torch.zeros(
            episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)
        self.rewards = torch.zeros(episode_length, n_rollout_threads, 1)
        self.value_preds = torch.zeros(episode_length + 1, n_rollout_threads, 1)
        self.returns = torch.zeros(episode_length + 1, n_rollout_threads, 1)
        self.action_log_probs = torch.zeros(episode_length, n_rollout_threads, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(episode_length, n_rollout_threads, action_shape)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.masks = torch.ones(episode_length + 1, n_rollout_threads, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(episode_length + 1, n_rollout_threads, 1)
        #我们设计的环境的特殊性，其实并不需要bad_masks

        self.episode_length = episode_length
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.inform = self.inform.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, inform,recurrent_hidden_states,actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.inform[self.step + 1].copy_(inform)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.episode_length


    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda
                        ):

        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (self.returns[step + 1] *
                                      gamma * self.masks[step + 1] + self.rewards[step])



    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):

        episode_length,n_rollout_threads=self.rewards.size()[0:2]
        batch_size = n_rollout_threads * episode_length
        #'''
        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch
        #'''

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            inform_batch = self.inform[:-1].view(-1, *self.inform.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, inform_batch ,recurrent_hidden_states_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

                
    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.size()[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length #[C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(data_chunks)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = []
            inform_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[L,Dim]
                obs_batch.append(self.obs[:episode_length].reshape(-1, *self.obs.size()[2:])[ind:ind+data_chunk_length])
                inform_batch.append(self.inform[:episode_length].reshape(-1, *self.inform.size()[2:])[ind:ind+data_chunk_length])
                actions_batch.append(self.actions[:episode_length].reshape(-1, self.actions.size(-1))[ind:ind+data_chunk_length])
                value_preds_batch.append(self.value_preds[:episode_length].reshape(-1, 1)[ind:ind+data_chunk_length])
                return_batch.append(self.returns[:episode_length].reshape(-1, 1)[ind:ind+data_chunk_length])
                masks_batch.append(self.masks[:episode_length].reshape(-1, 1)[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(self.action_log_probs[:episode_length].reshape(-1, 1)[ind:ind+data_chunk_length])
                adv_targ.append(advantages.reshape(-1, 1)[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[:episode_length].reshape(
                      -1, self.recurrent_hidden_states.size(-1))[ind])
                      
            L, N =  data_chunk_length, mini_batch_size

            # These are all tensors of size (L, N, Dim)
            obs_batch = torch.stack(obs_batch).transpose(0,1).reshape(-1,*self.obs.size()[2:])
            inform_batch = torch.stack(inform_batch).transpose(0,1).reshape(-1, *self.inform.size()[2:])
            actions_batch = torch.stack(actions_batch).transpose(0,1).reshape(-1, self.actions.size(-1))
            value_preds_batch = torch.stack(value_preds_batch).transpose(0,1).reshape(-1, 1)
            return_batch = torch.stack(return_batch).transpose(0,1).reshape(-1, 1)
            masks_batch = torch.stack(masks_batch).transpose(0,1).reshape(-1, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch).transpose(0,1).reshape(-1, 1)
            adv_targ = torch.stack(adv_targ).transpose(0,1).reshape(-1, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch)#.view(N, -1)


            
            yield obs_batch,inform_batch, recurrent_hidden_states_batch,\
                  actions_batch, value_preds_batch, return_batch, masks_batch, \
                  old_action_log_probs_batch, adv_targ
            
