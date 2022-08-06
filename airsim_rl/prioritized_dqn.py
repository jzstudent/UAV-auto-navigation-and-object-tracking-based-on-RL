import math, random
import gym
import numpy as np
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import gym_airsim
from game_handling.game_handler_class import *
from Rainbow.common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from tqdm import trange
import cv2


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

class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)


    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0


        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def push_airsim(self, state, action, reward, next_state, done):
        img=state[0]
        inform=state[1]
        state = [img,inform]
        img = next_state[0]
        inform = next_state[1]
        next_state = [img,inform]

        max_prio = self.priorities.max() if self.buffer else 1.0


        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs  = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        batch       = list(zip(*samples))
        states      = batch[0]
        actions     = batch[1]
        rewards     = batch[2]
        next_states = batch[3]
        dones       = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size()+10, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon,device):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            #action = q_value.max(1)[1].data[0]
            action = q_value.max(1)[1].cpu().data[0].numpy()
        else:
            action = np.random.choice(self.num_actions)
        return action


    def act_airsim(self, state, epsilon,device):
        if random.random() > epsilon:

            if len(state)==2:
                state=[state]

            img = np.array([i[0] for i in state])
            inform = np.array([i[1] for i in state])


            img = img / 255.0
            img = torch.FloatTensor(np.float32(img)).to(device)
            inform = torch.FloatTensor(np.float32(inform)).to(device)
            ####### TO:(N,C,H,W)
            #img = img.permute(0, 3, 1, 2)

            feature=self.features(img)
            feature = feature.view(feature.size(0), -1)
            x=torch.cat((feature,inform),-1)
            q_value = self.fc(x)
            #action = q_value.max(1)[1].data[0]
            action = q_value.max(1)[1].cpu().data[0].numpy()
        else:
            action = np.random.choice(self.num_actions)
        return action

    def forward_airsim(self,state,device):
        if len(state) == 2:
            state = [state]

        img = np.array([i[0] for i in state])
        inform = np.array([i[1] for i in state])

        img = img / 255.0
        img = torch.FloatTensor(np.float32(img)).to(device)
        inform = torch.FloatTensor(np.float32(inform)).to(device)
        ####### TO:(N,C,H,W)
        #img = img.permute(0, 3, 1, 2)

        feature = self.features(img)
        feature = feature.view(feature.size(0), -1)
        x = torch.cat((feature, inform), -1)
        q_value = self.fc(x)
        return q_value


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_inputs= num_inputs
        self.num_actions= num_actions
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon,device):
        if random.random() > epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].cpu().data[0].numpy()
        else:
            #action = random.randrange(self.num_actions)
            action = np.random.choice(self.num_actions)
        return action


class rew_bn(nn.Module):
    def __init__(self,beta=0.9999,device="cuda:0"):
        super(rew_bn, self).__init__()

        self.mu=nn.Parameter(data=torch.zeros(1),requires_grad=True)
        self.sigma=nn.Parameter(data=torch.ones(1),requires_grad=True)

        self.meaning=torch.zeros(1).to(device)
        self.sq_meaning = torch.zeros(1).to(device)
        self.beta=beta
        self.epsilon=1e-2
        #self.bnc=nn.BatchNorm1d(1)

    def forward(self,rew):
        #'''

        batch_mean=rew.mean()
        batch_sq_mean=batch_mean**2

        self.meaning.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.sq_meaning.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        var = (self.sq_meaning - self.meaning ** 2).clamp(min=self.epsilon)

        rew=(rew-self.meaning)/torch.sqrt(var)

        rew= rew*self.sigma+self.mu
        #'''
        #rew=self.bnc(rew.unsqueeze(-1)).squeeze(-1)
        return rew



class PER_DQN(object):

    def __init__(self, num_inputs, num_actions,lr,eps,capacity,CNN=False,REW_BN=False,device="cpu"):
        #
        if CNN:
            self.current_model = CnnDQN(num_inputs, num_actions)
            self.target_model = CnnDQN(num_inputs, num_actions)
        else:
            self.current_model = DQN(num_inputs[0], num_actions)
            self.target_model = DQN(num_inputs[0], num_actions)

        self.device=device
        self.optimizer = optim.Adam(self.current_model.parameters(),lr=lr,eps=eps)

        self.replay_buffer = NaivePrioritizedBuffer(capacity)
        if REW_BN:
            self.bn=rew_bn()
            self.bn_optimizer= optim.Adam(self.bn.parameters(),lr=lr,eps=eps)

    def compute_td_loss(self,batch_size, beta):

        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta)

        #state = torch.FloatTensor(np.float32(state)).to(self.device)
        #next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        if REW_BN:
            reward=self.bn(reward)

        q_values = current_model.forward_airsim(state,self.device)
        next_q_values = target_model.forward_airsim(next_state,self.device)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + (gamma * next_q_value * (1 - done)).detach()

        loss = (q_value - expected_q_value).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        if REW_BN:
            self.bn_optimizer.zero_grad()

        loss.backward()

        _, grad_norm = get_p_and_g_mean_norm(self.current_model.parameters())

        nn.utils.clip_grad_norm_(self.current_model.parameters(), 20)

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()
        if REW_BN:
            self.bn_optimizer.step()
        return loss,grad_norm

    def soft_update(self,tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(self.target_model.parameters(), self.current_model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    def hard_update(self):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(self.target_model.parameters(), self.current_model.parameters()):
            target_param.data.copy_(param.data)


def start_game():
    game_handler = GameHandler()
    game_handler.start_game_in_editor()





seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# cuda
if torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    torch.set_num_threads(8)

env_id = "AirSimEnv-v42"#CartPole-v0 "AirSimEnv-v42" PongNoFrameskip-v4
if env_id =="AirSimEnv-v42":
    start_game()
    env = gym.make(env_id)
    env.seed(seed)
elif env_id =="CartPole-v0":
    env = gym.make(env_id)
    env.seed(seed)
elif env_id=="PongNoFrameskip-v4":
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

# path
model_dir = Path('./results') / env_id / "per"
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
logger = SummaryWriter(str(log_dir))


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device="cuda:0"
else:
    device="cpu"
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

beta_start = 0.4
beta_frames = 100000#100000#1000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000#30000#500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


lr=1e-4
eps=1e-5
capacity=100000
tau=0.0001


REW_BN=False

per_dqn=PER_DQN(env.observation_space.shape, env.action_space.n,lr,eps,capacity,CNN=True,REW_BN=REW_BN,device=device)


current_model = per_dqn.current_model.to(device)
target_model  = per_dqn.target_model.to(device)
if REW_BN:
    per_dqn.bn.to(device)
per_dqn.hard_update()


num_frames = 1000000
batch_size = 64
gamma = 0.99
losses = []
all_rewards = []
episode_reward = 0
step=0
state = env.reset()

for frame_idx in trange(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = per_dqn.current_model.act_airsim(state, epsilon,device)

    ##in airsim
    if env_id == "AirSimEnv-v42":
       next_state, reward, done, _ = env.step([action])
    else:
        next_state, reward, done, _ = env.step(action)

    reward=reward/50

    per_dqn.replay_buffer.push_airsim(state, action, reward, next_state, done)

    img = np.array(state[0][0], dtype=np.uint8)
    cv2.imshow("0", img)
    cv2.waitKey(1)

    state = next_state
    episode_reward += reward

    if done:
        step+=1
        state = env.reset()
        all_rewards.append(episode_reward)
        logger.add_scalars('rews',
                           {'rews': episode_reward},
                           step)
        episode_reward = 0


    if len(per_dqn.replay_buffer) > 1000:#batch_size:
        beta = beta_by_frame(frame_idx)
        loss,grad_norm = per_dqn.compute_td_loss(batch_size, beta)
        losses.append(loss.data)

        logger.add_scalars('grad_norm',
                                {'grad_norm': grad_norm},
                                frame_idx)
        logger.add_scalars('value_loss',
                           {'value_loss': loss},
                           frame_idx)

        #per_dqn.soft_update(tau)

    if frame_idx % 1000 == 0:
        per_dqn.hard_update()

    if (frame_idx % 10000 == 0 or frame_idx == num_frames - 1):
        torch.save({
            'model': per_dqn.current_model
        },
            str(save_dir) + "/current_model" + ".pt")