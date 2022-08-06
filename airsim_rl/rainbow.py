import gym
import numpy as np
import random,math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from Rainbow.common.layers import NoisyLinear
import gym_airsim
from game_handling.game_handler_class import *
from tqdm import trange
from pathlib import Path
from tensorboardX import SummaryWriter


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
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        #assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

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
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = np.array(batch[2])
        next_states = np.concatenate(batch[3])
        dones       = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)


        x = value + advantage - advantage.mean(1, keepdim=True)

        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
    '''
    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        action = dist.sum(2).max(1)[1].numpy()[0]
        return action
    '''
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            dist = self.forward(state).data.cpu()
            dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
            action = dist.sum(2).max(1)[1].numpy()[0]
        else:
            action = random.randrange(env.action_space.n)
        return action
    #'''

class Rainbow(object):

    def __init__(self,lr,num_inputs, num_actions, num_atoms, Vmin, Vmax,capacity,use_popart, prob_alpha=0.6):
        self.current_model = RainbowDQN(num_inputs, num_actions, num_atoms, Vmin, Vmax)
        self.target_model = RainbowDQN(num_inputs, num_actions, num_atoms, Vmin, Vmax)

        self.replay_buffer=NaivePrioritizedBuffer(capacity,prob_alpha)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr)
        self.num_atoms=num_atoms
        self.Vmin=Vmin
        self.Vmax=Vmax

        self.use_popart=use_popart
        if use_popart:
            self.popart = PopArt(1)


    def soft_update(self, tau):
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

    def projection_distribution(self,next_state, rewards, dones):
        batch_size = next_state.size(0)


        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)


        next_dist = target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        next_dist_real = target_model(next_state).data.cpu()
        next_dist_real = next_dist_real.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z

        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())

        #proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        #proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist_real * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist_real * (b - l.float())).view(-1))


        proj_dist=proj_dist/proj_dist.sum(1).unsqueeze(1)

        return proj_dist


    def compute_td_loss(self,batch_size,beta):
        #state, action, reward, next_state, done = self.replay_buffer.sample(batch_size,beta)
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta)


        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))
        weights = Variable(torch.FloatTensor(weights))


        proj_dist = self.projection_distribution(next_state, reward, done)

        dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(Variable(proj_dist) * dist.log()).sum(1)

        prios = loss*weights + 1e-5
        loss = sum(loss*weights)/len(loss*weights)

        self.optimizer.zero_grad()
        loss.backward()

        _, grad_norm = get_p_and_g_mean_norm(self.current_model.parameters())
        #nn.utils.clip_grad_norm_(self.current_model.parameters(), 100)

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        current_model.reset_noise()
        target_model.reset_noise()

        return loss,grad_norm

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

#start_game()
#env_id = "AirSimEnv-v42"
env_id = "CartPole-v0"
env = gym.make(env_id)

# path
model_dir = Path('./results') / env_id / "rainbow"
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


tau=0.01
lr=5e-4
num_atoms = 51
Vmin = -30
Vmax = 30
capacity=100000
num_frames = 100000
batch_size = 64
gamma = 0.99

beta_start = 0.4
beta_frames = 1000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

use_popart=True
rainbow= Rainbow(lr,env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax,capacity,use_popart)
if USE_CUDA:
    current_model = rainbow.current_model.cuda()
    target_model = rainbow.target_model.cuda()

rainbow.hard_update()



step=0
state = env.reset()
for frame_idx in trange(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = rainbow.current_model.act(state, epsilon)

    ##in airsim
    if env_id == "AirSimEnv-v42":
        next_state, reward, done, _ = env.step([action])
        reward=reward/100
    else:
        next_state, reward, done, _ = env.step(action)
    rainbow.replay_buffer.push(state, action, reward, next_state, done)

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


    if len(rainbow.replay_buffer) > batch_size:
        beta = beta_by_frame(frame_idx)
        loss,grad_norm = rainbow.compute_td_loss(batch_size, beta)
        losses.append(loss.data)

        logger.add_scalars('grad_norm',
                                {'grad_norm': grad_norm},
                                frame_idx)
        logger.add_scalars('value_loss',
                           {'value_loss': loss},
                           frame_idx)

        rainbow.soft_update(tau)

    #if frame_idx % 2000 == 0:
    #    per_dqn.hard_update()

    if (frame_idx % 10000 == 0 or frame_idx == num_frames - 1):
        torch.save({
            'model': rainbow.current_model
        },
            str(save_dir) + "/current_model" + ".pt")