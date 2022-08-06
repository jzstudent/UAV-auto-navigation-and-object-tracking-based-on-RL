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
from tensorboardX import SummaryWriter
from tqdm import trange
from pathlib import Path
from torch.distributions import Categorical,Independent
import cv2
import numpy as np



INCORPORATE = 7
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.inform_buf = np.zeros(combined_shape(size, INCORPORATE), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.inform2_buf = np.zeros(combined_shape(size, INCORPORATE), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs[0]
        self.inform_buf[self.ptr] = obs[1]
        self.obs2_buf[self.ptr] = next_obs[0]
        self.inform2_buf[self.ptr] = next_obs[1]
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, device=torch.device("cpu")):
        idxs = np.random.choice(np.arange(self.size), size=batch_size, replace=False)
        batch = dict(obs=[torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32).to(device),
                          torch.as_tensor(self.inform_buf[idxs], dtype=torch.float32).to(device)],
                     obs2=[torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32).to(device),
                           torch.as_tensor(self.inform2_buf[idxs], dtype=torch.float32).to(device)],
                     act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).to(device),
                     rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device),
                     done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device))
        return {k: v for k, v in batch.items()}



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


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

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
                 sattn = True,
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


def sac(device, seed=1, total_steps=int(510000), replay_size=int(150000), gamma=0.99,
        polyak=0.995, lr=5e-4, alpha=0.2, batch_size=256, start_steps=5000,
        update_after=10000, update_every=50, save_freq=3000, sattn= True):
    ##310000-->410000  1e5--->3e5
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = AirSimEnv(need_render=False)
    #env = gym.make("HalfCheetah-v2")
    env.seed(seed)

    model_dir = Path('./results') / 'AirSimEnv-v42'/ 'SAC'
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    best_sr=0

    # Create actor-critic module and target networks
    ac = SACActorCritic(env.observation_space, env.action_space, device = device, sattn=sattn)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    ac_target = deepcopy(ac)
    for p in ac_target.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(),ac.q2.parameters())
    # Experience buffer
    obs_dim = env.observation_space.shape

    if env.action_space.__class__.__name__ == "Box":
        act_dim = env.action_space.shape[0]
        discrete = False

    else:
        act_dim = 1
        discrete = True

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        if discrete:
            with torch.no_grad():
                dist,probs,_ = ac.pi.evaluate(o2)
                q1_targ = ac_target.q1.evaluate(o2)
                q2_targ = ac_target.q2.evaluate(o2)
                q_target = torch.min(q1_targ,q2_targ)
                v = (probs*q_target).sum(dim=-1)+alpha*dist.entropy()

                backup = r + gamma * (1 - d) *v

        else:

            with torch.no_grad():
                # Target actions come from *current* policy
                a2, logp_a2 = ac.pi(o2)

                # Target Q-values
                q1_pi_targ = ac_target.q1(o2, a2)
                q2_pi_targ = ac_target.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ,q2_pi_targ)
                backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1+loss_q2

        return loss_q, dist.entropy().mean()

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']

        if discrete:
            _, probs, log_prob = ac.pi.evaluate(o)

            q1 = ac.q1.evaluate(o)
            q2 = ac.q2.evaluate(o)

            q_min = torch.min(q1, q2)
            loss_pi = (probs * (-q_min + alpha * log_prob)).sum(dim=-1).mean()
        else:
            pi, logp_pi = ac.pi(o)
            q1_pi = ac.q1(o,pi)
            q2_pi = ac.q2(o, pi)
            q_pi = torch.min(q1_pi,q2_pi)

            # Entropy-regularized policy loss
            loss_pi = (alpha * logp_pi - q_pi).mean()

        return loss_pi

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)


    def update(data,logger,t):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, entropy = compute_loss_q(data)
        logger.add_scalars('value_loss',
                                {'value_loss': loss_q.item()},
                                t)
        loss_q.backward()
        '''
        norm, grad_norm = get_p_and_g_mean_norm(ac.q1.parameters())
        logger.add_scalars('q1_grad_norm',
                           {'q1_grad_norm': grad_norm},
                           t)
        norm, grad_norm = get_p_and_g_mean_norm(ac.q2.parameters())
        logger.add_scalars('q2_grad_norm',
                           {'q2_grad_norm': grad_norm},
                           t)
        logger.add_scalars('entropy',
                           {'entropy': entropy.detach()},
                           t)
        '''
        #nn.utils.clip_grad_norm_(q_params, 5)
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        '''
        norm, grad_norm = get_p_and_g_mean_norm(ac.pi.parameters())
        logger.add_scalars('pi_grad_norm',
                           {'pi_grad_norm': grad_norm},
                           t)
        '''
        #nn.utils.clip_grad_norm_(ac.pi.parameters(), 5)

        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act([torch.as_tensor(o[0], dtype=torch.float32).unsqueeze(0).to(device),
                            torch.as_tensor(o[1], dtype=torch.float32).unsqueeze(0).to(device)],
                      deterministic)

    # Prepare for interaction with environment
    logger = SummaryWriter(str(log_dir))
    o = env.reset()
    ep_ret = 0
    ep_len = 0
    num_epi = 0
    rews_deque = collections.deque(maxlen=100)
    steps_deque = collections.deque(maxlen=100)
    success_deque = collections.deque(maxlen=100)
    # Main loop: collect experience in env and update/log each epoch
    for t in trange(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.

        if t > start_steps:
            a = get_action(o)
            if discrete:
                a=a[0]

        else:
            a = env.action_space.sample()
            if discrete:
                a = np.array([a])
            else:
                a=a[None,::]


        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if env.stepN==1 and d:
            print("WTF!")
            pass
        else:
            if discrete:
                replay_buffer.store(o, a, r, o2, d)
            else:
                replay_buffer.store(o,a[0],r,o2,d)
        img = o[0]
        img = np.hstack(img)
        img = np.array(img, dtype=np.uint8)
        # img = np.array(obs[0][0] , dtype=np.uint8)
        cv2.imshow("0", img)
        cv2.waitKey(1)
        o = o2

        #env.airgym.client.simPause(True)
        if d:
            num_epi += 1
            if env.success:
                success_deque.append(1)
            else:
                success_deque.append(0)
            o = env.reset()
            rews_deque.append(ep_ret)
            steps_deque.append(ep_len)
            logger.add_scalars('mean_episode_reward',
                               {'mean_episode_reward': sum(rews_deque) / len(rews_deque)},
                               num_epi)
            logger.add_scalars('mean_episode_length',
                               {'mean_episode_reward': sum(steps_deque) / len(steps_deque)},
                               num_epi)
            logger.add_scalars('success_rate',
                               {'success_rate': sum(success_deque) / len(success_deque)},
                               num_epi)
            ep_len=0
            ep_ret=0

        if t>= update_after and t % update_every == 0:
            for j in range(update_every//5):
                batch = replay_buffer.sample_batch(batch_size, device)
                update(batch,logger,t)


        if t % save_freq==0 or t ==total_steps-1:
            torch.save({
                'model': ac
            },
                str(save_dir) +"/actor_model_{}".format(t) + ".pt")
        #env.airgym.client.simPause(False)
        if d and sum(success_deque) / len(success_deque)>best_sr:
            torch.save({
                'model': ac
            },
                str(save_dir) + "/actor_model_best" + ".pt")
            best_sr=sum(success_deque) / len(success_deque)

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