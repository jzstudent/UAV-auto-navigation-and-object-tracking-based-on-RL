from copy import deepcopy
import itertools
from torch.optim import Adam
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from common.utils import *
from gym_airsim.envs.AirGym import AirSimEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from tqdm import trange
from pathlib import Path
from torch.distributions import Categorical,Independent
import cv2
INCORPORATE = 7
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def plot_attention(attention, name):

    d = np.array(attention)
    col = [t for t in range(d.shape[0])]
    index = [t for t in range(d.shape[1])]
    df = pd.DataFrame(d, columns=col, index=index)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))

    plt.savefig(name)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class CNNAttention(nn.Module):
    def __init__(self,size, name ="sa"):
        super(CNNAttention,self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.w_qs=init_(nn.Conv2d(size, size//8, 1))
        self.w_ks=init_(nn.Conv2d(size, size//8, 1))
        self.w_vs=init_(nn.Conv2d(size, size, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.name = name

    def forward(self,inputs):

        Batch=inputs.shape[0]
        output_size=inputs.shape[2]*inputs.shape[3]
        # NCHW-->N,C,H*W-->N,H*W,C
        q=self.w_qs(inputs).view(Batch,-1,output_size).permute(0,2,1) # N,H*W,C

        k = self.w_ks(inputs).view(Batch,-1,output_size)# N,C,H*W

        v = self.w_vs(inputs).view(Batch,-1,output_size)# N,C,H*W

        attn = torch.bmm(q, k) #N,H*W,H*W

        attn=F.softmax(attn,dim=-1)
        #plot_attention(attn[0].sum(dim=0).view(inputs.shape[2],inputs.shape[3]).detach().cpu().numpy(),
        #               self.name)

        # N,C,H*W
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out=out.view(*inputs.shape)

        out= self.gamma*out + inputs

        return out

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size, activation, act_limit=None,
                 discrete=False, sattn = False):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        num_inputs = obs_dim[0]
        self.discrete = discrete
        if sattn:
            self.layer1 = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),
                CNNAttention(16, name = "ac_l1"),
                init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),
                CNNAttention(32, name = "ac_l2"),
                init_(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 1, stride=1)), nn.ReLU(),
                Flatten(),
            )
        else:
            self.layer1 = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 1, stride=1)), nn.ReLU(),
                Flatten(),
            )
        self.layer2 = nn.Sequential(init_(nn.Linear(INCORPORATE, 64)),
                                    nn.ReLU(),
                                    )

        if discrete:
            self.prob = nn.Sequential(init_(nn.Linear(800 + 64, hidden_size)),
                                    activation(),
                                    init_(nn.Linear(hidden_size, act_dim))
                                    )

        else:
            self.mu = nn.Sequential(init_(nn.Linear(800 + 64, hidden_size)),
                                        activation(),
                                        init_(nn.Linear(hidden_size, act_dim))
                                        )

            self.log_std = nn.Sequential(init_(nn.Linear(800 + 64, hidden_size)),
                                    activation(),
                                    init_(nn.Linear(hidden_size, act_dim))
                                    )

            self.act_limit=act_limit

        # self.train()

    def evaluate(self,obs):
        img = obs[0]
        inform = obs[1]

        x = img / 255.0

        #######TO:(N,C,H,W)
        x = self.layer1(x)
        inform = self.layer2(inform)
        x = torch.cat((x, inform), -1)

        x = self.prob(x)

        distribution = Categorical(logits=x)

        action_probs = F.softmax(x,dim=-1)
        z = (action_probs==0.0).float()*1e-8
        log_action_probs = torch.log(action_probs+z)

        return distribution, action_probs, log_action_probs

    def forward(self, obs, deterministic=False, with_logprob=True):

        if self.discrete:
            img = obs[0]
            inform = obs[1]

            x = img / 255.0

            #######TO:(N,C,H,W)
            x = self.layer1(x)
            inform = self.layer2(inform)
            x = torch.cat((x, inform), -1)

            x = self.prob(x)

            distribution = Categorical(logits=x)

            if deterministic:
                action = distribution.probs.argmax(dim=-1)
            else:
                action = distribution.sample()

            if with_logprob:
                action_log_probs = distribution.log_prob(action)
            else:
                action_log_probs = None

            return action.unsqueeze(-1), action_log_probs

        else:
            img = obs[0]
            inform = obs[1]

            x = img / 255.0

            #######TO:(N,C,H,W)
            x = self.layer1(x)
            inform = self.layer2(inform)
            x = torch.cat((x, inform), -1)

            mu = self.mu(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)

            pi_dis = Independent(Normal(mu, std),1)

            if deterministic:
                pi_action = mu
            else:
                pi_action = pi_dis.rsample()

            if with_logprob:
                log_pi = pi_dis.log_prob(pi_action)
                log_pi -= (2*(np.log(2)-pi_action-F.softplus(-2*pi_action))).sum(axis=1)

            else:
                log_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit*pi_action

            return pi_action, log_pi

class Qfunc(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, activation, discrete=False,
                 sattn=False):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        num_inputs = obs_dim[0]
        self.discrete = discrete
        if sattn:
            self.layer1 = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),
                CNNAttention(16, name = "q_l1"),
                init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),
                CNNAttention(32, name = "q_l2"),
                init_(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 1, stride=1)), nn.ReLU(),
                Flatten(),
            )
        else:
            self.layer1 = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 1, stride=1)), nn.ReLU(),
                Flatten(),
            )

        if discrete:
            self.layer2 = nn.Sequential(init_(nn.Linear(INCORPORATE, 64)),
                                        nn.ReLU(),
                                        )
            self.q = nn.Sequential(init_(nn.Linear(64+ 800, hidden_size)),
                                    activation(),
                                    init_(nn.Linear(hidden_size, act_dim))
                                    )
        else:
            self.layer2 = nn.Sequential(init_(nn.Linear(INCORPORATE + 800, hidden_size)),
                                        nn.ReLU(),
                                        )
            self.q = nn.Sequential(init_(nn.Linear(hidden_size + act_dim, hidden_size)),
                                    activation(),
                                    init_(nn.Linear(hidden_size, 1))
                                    )

    def evaluate(self, obs):
        img = obs[0]
        inform = obs[1]

        x = img / 255.0

        #######TO:(N,C,H,W)
        x = self.layer1(x)
        inform = self.layer2(inform)
        x = torch.cat((x, inform), -1)

        q = self.q(x)

        return q

    def forward(self, obs, act):

        if self.discrete:
            img = obs[0]
            inform = obs[1]

            x = img / 255.0

            #######TO:(N,C,H,W)
            x = self.layer1(x)
            inform = self.layer2(inform)
            x = torch.cat((x, inform), -1)


            q = self.q(x).gather(1,act.long())

            return q.squeeze(-1)
        else:
            img = obs[0]
            inform = obs[1]

            x = img / 255.0

            #######TO:(N,C,H,W)
            x = self.layer1(x)

            x = torch.cat((x, inform), -1)

            x = self.layer2(x)

            x = torch.cat((x, act), -1)

            q = self.q(x)

            return q.squeeze(-1)

class SACActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 sattn = False,
                 hidden_size=512,
                 activation=nn.ReLU,
                 ):
        super().__init__()
        obs_dim=observation_space.shape

        if action_space.__class__.__name__ == "Box":
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
            self.pi = Actor(obs_dim, act_dim, hidden_size, activation, act_limit, sattn=sattn).to(device)
            self.q1 = Qfunc(obs_dim, act_dim, hidden_size, activation, sattn=sattn).to(device)
            self.q2 = Qfunc(obs_dim, act_dim, hidden_size, activation, sattn=sattn).to(device)

        else:
            act_dim = action_space.n
            self.pi = Actor(obs_dim, act_dim, hidden_size, activation, discrete=True, sattn=sattn).to(device)
            self.q1 = Qfunc(obs_dim, act_dim, hidden_size, activation, discrete=True, sattn=sattn).to(device)
            self.q2 = Qfunc(obs_dim, act_dim, hidden_size, activation, discrete=True, sattn=sattn).to(device)


    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()


def sac(device, seed=100, total_steps=int(5000), replay_size=int(1e5), gamma=0.99,
        polyak=0.995, lr=5e-4, alpha=0.2, batch_size=256, start_steps=5000,
        update_after=10000, update_every=50, save_freq=3000, sattn= True):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = AirSimEnv(need_render=False)
    env.seed(seed)

    model_dir = Path('./results') / 'AirSimEnv-v42'/ 'SAC'/ 'run7'/'models'
    ac = torch.load(str(model_dir) + "/actor_model_507000" + ".pt")['model']
    ac.to(device)
    print(ac)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac.parameters():
        p.requires_grad = False

    if env.action_space.__class__.__name__ == "Box":
        discrete = False

    else:
        discrete = True

    def get_action(o, deterministic=False):
        return ac.act([torch.as_tensor(o[0], dtype=torch.float32).unsqueeze(0).to(device),
                            torch.as_tensor(o[1], dtype=torch.float32).unsqueeze(0).to(device)],
                      deterministic)

    # Prepare for interaction with environment
    o = env.reset()
    ep_ret = 0
    ep_len = 0
    # Main loop: collect experience in env and update/log each epoch
    for t in trange(total_steps):


        a = get_action(o,True)
        if discrete:
            a=a[0]


        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        img = o[0]
        img = np.hstack(img)
        img = np.array(img, dtype=np.uint8)
        # img = np.array(obs[0][0] , dtype=np.uint8)
        cv2.imshow("0", img)
        cv2.waitKey(1)
        o = o2


        if d:
            o = env.reset()
            print(ep_len,":",ep_ret)
            ep_len=0
            ep_ret=0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    args = parser.parse_args()

    #if torch.cuda.is_available():
    #    device = torch.device("cuda:1")
    #    torch.set_num_threads(1)
    #else:
    #    device = torch.device("cpu")

    device = torch.device("cuda:0")
    torch.set_num_threads(torch.get_num_threads())

    sac(device=device)