import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.distributions import Categorical, DiagGaussian
from utils.util import init


INCORPORATE=9

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()

        self.obs_shape=obs_shape
        if base_kwargs is None:
            raise NotImplementedError
        if len(obs_shape) == 3:
            base = CNNBase
            self.base = base(obs_shape, **base_kwargs)
        elif len(obs_shape) == 1:
            base = MLPBase
            self.base = base(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError


        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)

        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)

        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs =self.base( inputs, rnn_hxs, masks)
        
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        #dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self,inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs= self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self,recurrent, recurrent_input_size, recurrent_hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = recurrent_hidden_size
        self._recurrent = recurrent

        if recurrent :
            self.gru = nn.GRU(recurrent_input_size, recurrent_hidden_size)
            #self.gru_critic = nn.GRU(recurrent_input_size, recurrent_hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)


    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x= self.gru(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)          
        else:
            # x is a (T * N, -1) tensor that has been flatten to (T, N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))
            
            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            has_zeros = ((masks[:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item()]
            else:
                has_zeros = has_zeros.numpy().tolist()

            if has_zeros:
                # add t=0 and t=T to the list
                if has_zeros[0]!=0:
                    has_zeros = [0] + has_zeros + [T]
                else:
                    has_zeros = has_zeros + [T]

                hxs = hxs.unsqueeze(0)

                outputs = []
                for i in range(len(has_zeros) - 1):
                    # We can now process steps that don't have any zeros in masks together!
                    # This is much faster
                    start_idx = has_zeros[i]
                    end_idx = has_zeros[i + 1]
                    rnn_scores, hxs = self.gru( x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))
                    outputs.append(rnn_scores)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.cat(outputs, dim=0)

                # flatten
                x = x.reshape(T * N, -1)
                hxs = hxs.squeeze(0)
            else:
                hxs = hxs.unsqueeze(0)
                rnn_scores, hxs = self.gru(x, hxs)

                # flatten
                x = rnn_scores.reshape(T * N, -1)
                hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self,
                 inputs,
                 recurrent=False,
                 hidden_size=64,
                 recurrent_input_size=64,
                 recurrent_hidden_size=64
                 ):
        super(CNNBase, self).__init__(recurrent, recurrent_input_size, recurrent_hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        num_inputs = inputs[0]

        self.layer1 = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 1, stride=1)), nn.ReLU(),
            Flatten(),
        )


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                       constant_(x, 0))

        self.layer2 = nn.Sequential(init_(nn.Linear(INCORPORATE, 64)),
                                    nn.ReLU(),
                                    )
        self.layer3 = nn.Sequential(init_(nn.Linear(800 + 64, recurrent_input_size)),
                                    nn.ReLU(),
                                    )
        if recurrent:

            self.critic_linear = nn.Sequential(init_(nn.Linear(recurrent_hidden_size, hidden_size)),
                                                nn.ReLU(),
                                                init_(nn.Linear(hidden_size, 1))
                                )

        else:
            self.critic_linear = nn.Sequential(init_(nn.Linear(recurrent_input_size, hidden_size)),
                                                nn.ReLU(),
                                                init_(nn.Linear(hidden_size, 1)))#init_(nn.Linear(recurrent_input_size, 1))

        #self.train()

    def forward(self, inputs, rnn_hxs, masks):

        img=inputs[0]
        inform=inputs[1]

        x = img / 255.0
        #inform = inform/100.0
        #######TO:(N,C,H,W)

        x = self.layer1(x)
        inform = self.layer2(inform)

        x=torch.cat((x,inform),-1)

        x = self.layer3(x)

        if self.is_recurrent :
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)


        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self,
                 num_inputs,
                 recurrent=False,
                 hidden_size=64,
                 recurrent_input_size=64,
                 recurrent_hidden_size=64):
        super(MLPBase, self).__init__(recurrent, recurrent_input_size, recurrent_hidden_size)

        if recurrent :
            num_inputs = recurrent_hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):

        x = inputs

        if self.is_recurrent :
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
