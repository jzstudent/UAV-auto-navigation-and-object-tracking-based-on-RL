import math, random
from gym_airsim.envs.AirGym import AirSimEnv
import gym
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from pathlib import Path
from collections import deque
import os
from tqdm import trange
INCORPORATE=9

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        s = np.expand_dims(state[0], 0)
        inform=np.expand_dims(state[1],0)
        next_s= np.expand_dims(next_state[0], 0)
        next_inform = np.expand_dims(next_state[1], 0)
        state=[s,inform]
        next_state=[next_s,next_inform]

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        s=[i[0] for i in state]
        inform=[i[1] for i in state]
        next_s=[i[0] for i in next_state]
        next_inform = [i[1] for i in next_state]

        return [np.concatenate(s),np.concatenate(inform)], action, reward, \
               [np.concatenate(next_s),np.concatenate(next_inform)], done

    def __len__(self):
        return len(self.buffer)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, stride=1),
            nn.ReLU(),
            Flatten(),
        )
        self.layer2 = nn.Sequential(nn.Linear(INCORPORATE, 64),
                                    nn.ReLU(),
                                    )
        self.q = nn.Sequential(
            nn.Linear(864, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        state=x[0]
        inform=x[1]
        state = state/255.0

        x = self.layer1(state)
        inform = self.layer2(inform)

        x=torch.cat((x,inform),-1)
        x = self.q(x)
        return x


    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = [Variable(torch.as_tensor(np.float32(state[0]), dtype=torch.float32)).unsqueeze(0),
                     Variable(torch.as_tensor(np.float32(state[1]), dtype=torch.float32)).unsqueeze(0)]
            with torch.no_grad():
                q_value = self.forward(state)
            action = q_value.max(1)[1].cpu().numpy()[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = [Variable(torch.as_tensor(np.float32(state[0]), dtype=torch.float32)),
             Variable(torch.as_tensor(np.float32(state[1]), dtype=torch.float32))]
    next_state = [Variable(torch.as_tensor(np.float32(next_state[0]), dtype=torch.float32)),
                  Variable(torch.as_tensor(np.float32(next_state[1]), dtype=torch.float32))]

    action = Variable(torch.as_tensor(action).long())
    reward = Variable(torch.as_tensor(reward, dtype=torch.float32))
    done = Variable(torch.as_tensor(done, dtype=torch.float32))
    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.smooth_l1_loss(q_value, Variable(expected_q_value.data).detach())

    #loss = (q_value - Variable(expected_q_value.data).detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    for param in current_model.parameters():
        param.grad.data.clamp_(-10, 10)
    optimizer.step()

    return loss

def soft_update(current_model, target_model, tau):
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


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(current_model, target_model):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target_model.parameters(), current_model.parameters()):
        target_param.data.copy_(param.data)


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
                    if USE_CUDA else autograd.Variable(*args, **kwargs)

env = AirSimEnv(need_render=False)
model_dir = Path('./results') / 'AirSimEnv-v42'/ 'dqn'
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


if USE_CUDA:
    device = torch.device("cuda:0")
    current_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
else:
    current_model = CnnDQN(env.observation_space.shape, env.action_space.n)
    target_model = CnnDQN(env.observation_space.shape, env.action_space.n)

target_model.load_state_dict(current_model.state_dict())
target_model.eval()

optimizer = optim.Adam(current_model.parameters(), lr=0.0005)



tua=0.01
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 100000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
replay_initial = 5000
replay_buffer = ReplayBuffer(100000)


#hard_update(current_model, target_model)

num_frames = 300000
batch_size = 256
gamma = 0.99


ep_ret = 0
ep_len = 0
num_epi = 0
rews_deque = deque(maxlen=100)
steps_deque = deque(maxlen=100)
success_deque = deque(maxlen=100)
save_freq=6000

state = env.reset()
for frame_idx in trange(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)

    next_state, reward, done, _ = env.step([action])
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    ep_ret += reward
    ep_len += 1

    if done:
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
        ep_len = 0
        ep_ret = 0


    if len(replay_buffer) > replay_initial and frame_idx % 50 == 0:
        print("update at ",frame_idx)
        for i in range(50):
            loss = compute_td_loss(batch_size)
            logger.add_scalars('value_loss',
                                    {'value_loss': loss.detach().cpu().numpy()},
                                    frame_idx)
            soft_update(current_model, target_model,tua)

    #if frame_idx % 1000 == 0:
    #    hard_update(current_model, target_model)


    if frame_idx % save_freq == 0 or frame_idx == num_frames - 1:
        torch.save({
            'model': current_model
        },
            str(save_dir) + "/actor_model_{}".format(frame_idx) + ".pt")
