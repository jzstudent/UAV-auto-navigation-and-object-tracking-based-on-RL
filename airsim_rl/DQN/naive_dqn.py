import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from collections import deque

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        self.buffer.append((state, action, reward, next_state , done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return np.array(state), np.array(action), np.array(reward),np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):

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

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)

        return state, action, reward, next_state, done, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class MLPDQN(nn.Module):
    def __init__(self,input_shape,num_actions,hidden,device):
        super(MLPDQN,self).__init__()
        self.input_shape=input_shape
        self.num_actions=num_actions
        self.device=device
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.layers = nn.Sequential(
            init_(nn.Linear(input_shape, hidden)),
            nn.ReLU(),
            init_(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            init_(nn.Linear(hidden, num_actions))
        )

    def forward(self,x):
        x=self.layers(x)
        return x

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_value = self.forward(state)
            action = q_value.max(1)[1].cpu().numpy()[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions,device):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device=device
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def feature_size(self):

        return self.features(torch.zeros(1, *self.input_shape).float()).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_value = self.forward(state)
            action = q_value.max(1)[1].cpu().numpy()[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

def compute_td_loss(replay_buffer,batch_size,device):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1).long()).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
    #loss = (q_value - Variable(expected_q_value.data).detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    #for param in current_model.parameters():
    #    param.grad.data.clamp_(-10, 10)
    optimizer.step()

    return loss


def compute_td_loss_per(replay_buffer,batch_size, beta,device):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size,beta)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action =torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)
    weights = torch.FloatTensor(weights).to(device)


    q_values = current_model(state)

    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1).long()).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss
def soft_update(current_model, target_model, tau=0.01):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target_model.parameters(), current_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



USE_CUDA = torch.cuda.is_available()
env = gym.make("PongNoFrameskip-v4")
from tensorboardX import SummaryWriter
logger = SummaryWriter("results/PongNoFrameskip-v4/dqn/logs")

epsilon_start = 0.9
epsilon_final = 0.005
epsilon_decay = 10000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

beta_start = 0.4
beta_frames = 100000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


if USE_CUDA:
    device = torch.device("cuda:0")
    #current_model = MLPDQN(env.observation_space.shape[0], env.action_space.n,128,device).to(device)
    #target_model = MLPDQN(env.observation_space.shape[0], env.action_space.n,128,device).to(device)
    ob_shape = (3,210,160)
    current_model = CnnDQN(ob_shape, env.action_space.n,device).to(device)
    target_model = CnnDQN(ob_shape, env.action_space.n,device).to(device)

else:
    device = torch.device("cpu")
    current_model = MLPDQN(env.observation_space.shape[0], env.action_space.n,128,device)
    target_model = MLPDQN(env.observation_space.shape[0], env.action_space.n,128,device)
target_model.load_state_dict(current_model.state_dict())
target_model.eval()

optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_initial = 5000
#replay_buffer = ReplayBuffer(1000)
replay_buffer = NaivePrioritizedBuffer(50000)

num_frames = 1000000
batch_size = 32
gamma = 0.99

episode_reward = 0
episode_nums=0
save_epis=1000

state = env.reset()
#state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
state = state.transpose(2, 0, 1)
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)

    action = current_model.act(state, epsilon)

    next_state, reward, done, _ = env.step(action)
    #next_state = cv2.resize(next_state, (84, 84), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    next_state = next_state.transpose(2, 0, 1)

    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()

        logger.add_scalars('rews',
                           {'rews': episode_reward},
                           episode_nums)
        print(episode_nums, episode_reward)
        episode_reward = 0
        episode_nums += 1


    if len(replay_buffer) > replay_initial:
        print("update at ",frame_idx)
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss_per(replay_buffer,batch_size,beta,device)
        logger.add_scalars('value_loss',
                                {'value_loss': loss},
                                frame_idx)

    #if frame_idx % 1000 == 0:
    #    target_model.load_state_dict(current_model.state_dict())

    soft_update(current_model, target_model)
    '''
    if episode_nums % save_epis == 0:
        torch.save({
            'model': current_model
        },
            "results/PongNoFrameskip-v4/dqn/model_{}".format(episode_nums) + ".pt")
    '''