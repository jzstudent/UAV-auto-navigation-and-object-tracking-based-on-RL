from copy import deepcopy
import itertools
from torch.optim import Adam
import collections

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from common.utils import *
from gym_airsim.envs.AirGym import AirSimEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from smoke.config import cfg
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from smoke.modeling.detector import build_detection_model
from smoke.modeling.heatmap_coder import get_transfrom_matrix
from smoke.structures.params_3d import ParamsList
from smoke.structures.image_list import to_image_list
import torch.multiprocessing as mp
from torch.multiprocessing import Array, Pipe

import torch
from torchvision.transforms import functional as Func
import os
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.patches as patches
from matplotlib.path import Path
import threading
from tqdm import trange
from pathlib import Path as path
from torch.distributions import Categorical,Independent
import cv2
INCORPORATE = 7
LOG_STD_MAX = 2
LOG_STD_MIN = -20

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}
VEHICLES=['Car', 'Cyclist', 'Pedestrian']
sim_mat=[286.81978,0.000000,245.011720,0.000000,
        0.000000, 281.38646,116.289441,0.000000,
        0.000000,0.000000,1.000000,0.000000]

tello_mat=[1891.67382,0.0000, 1239.04049,0.000000,
            0.0000000,1889.65445,8.71607713e+02,0.000000,
            0.000000,0.00000,1.00000,0.000000]

class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi


def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(ax, P2, line, color):

    corners_2D = compute_3Dbox(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)

def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor():
    def __call__(self, image, target):
        return Func.to_tensor(image), target


class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image, target):
        if self.to_bgr:
            image = image[[2, 1, 0]]
        image = Func.normalize(image, mean=self.mean, std=self.std)
        return image, target

def build_transforms(cfg):
    to_bgr = cfg.INPUT.TO_BGR

    normalize_transform = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=to_bgr
    )

    transform = Compose(
        [
            ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def img_process(img):
    # load default parameter here

    K=sim_mat
    K = np.array(K, dtype=np.float32).reshape(3, 4)
    K = K[:3, :3]

    center = np.array([i / 2 for i in img.size], dtype=np.float32)
    size = np.array([i for i in img.size], dtype=np.float32)


    center_size = [center, size]
    trans_affine = get_transfrom_matrix(
        center_size,
        [1280, 384]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)

    img = img.transform(
        (1280, 384),
        method=Image.AFFINE,
        data=trans_affine_inv.flatten()[:6],
        resample=Image.BILINEAR,
    )

    trans_mat = get_transfrom_matrix(
        center_size,
        [1280//4, 384//4]
    )

    # for inference we parametrize with original size
    target = ParamsList(image_size=size,
                        is_train=False)
    target.add_field("trans_mat", trans_mat)
    target.add_field("K", K)
    img, target = build_transforms(cfg)(img, target)

    return img,target

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


def sac(ac,device,goal_share, seed=100, total_steps=int(5000), replay_size=int(1e5), gamma=0.99,
        polyak=0.995, lr=5e-4, alpha=0.2, batch_size=256, start_steps=5000,
        update_after=10000, update_every=50, save_freq=3000, sattn= True):


    env = AirSimEnv(need_render=False)

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

    env.airgym.client.takeoffAsync().join()
    env.airgym.client.moveByVelocityZAsync(0, 0, env.airgym.z, 1).join()
    goal = env.airgym.client.simGetObjectPose("person")
    env.goal = [goal.position.x_val, goal.position.y_val, 0]
    init_pos = env.airgym.drone_pos()
    env.on_episode_start()
    print("done on episode start")

    state = env.init_state_f()
    env.prev_state = state
    o = state
    ep_ret = 0
    ep_len = 0


    # Main loop: collect experience in env and update/log each epoch
    test_goal=[1000,1000]
    kalman=cv2.KalmanFilter(4,2)
    kalman.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.15
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.measurementNoiseCov=np.array([[1,0],[0,1]],np.float32)*0.2
    frame=np.ones((500,500,3),np.uint8)*255
    current_measurement = np.zeros((2,1),np.float32)
    current_prediction = np.zeros((4, 1), np.float32)
    current_real =np.zeros((2,1),np.float32)
    while True:

        goal = env.airgym.client.simGetObjectPose("person")

        now = env.airgym.drone_pos()

        distance = np.sqrt(np.power(goal.position.x_val - now[0], 2)
                           + np.power(goal.position.y_val - now[1], 2)
                           )

        dis=(test_goal[0]-goal.position.x_val)**2\
              +(test_goal[1]-goal.position.y_val)**2
        for i in range(4):
            d=(goal_share[i*2]+now[0]-goal.position.x_val)**2\
              +(goal_share[i*2+1]+now[1]-goal.position.y_val)**2
            if d<dis:
                dis=d
                test_goal=[goal_share[i*2]+now[0],goal_share[i*2+1]+now[1]]
        print("test_goal",test_goal)

        last_measurement = current_measurement
        last_prediction = current_prediction
        last_real = current_real
        current_measurement=np.array((test_goal[0],test_goal[1]), np.float32)
        kalman.correct(current_measurement)
        current_prediction=kalman.predict()
        if goal.position.x_val<10000000 and goal.position.y_val<1000000:
            current_real = np.array([goal.position.x_val, goal.position.y_val], np.float32)

        lmx,lmy = int(last_measurement[0]-init_pos[0]+40)*4,int(last_measurement[1]-init_pos[1]+40)*4
        lpx, lpy = int(last_prediction[0]-init_pos[0]+40)*4, int(last_prediction[1]-init_pos[1]+40)*4
        lrx, lry = int(last_real[0]-init_pos[0]+40)*4, int(last_real[1]-init_pos[1]+40)*4

        cmx, cmy = int(current_measurement[0]-init_pos[0]+40)*4, int(current_measurement[1]-init_pos[1]+40)*4
        cpx, cpy = int(current_prediction[0]-init_pos[0]+40)*4, int(current_prediction[1]-init_pos[1]+40)*4
        crx, cry = int(current_real[0]-init_pos[0]+40)*4, int(current_real[1]-init_pos[1]+40)*4

        cv2.line(frame,(lmx,lmy),(cmx,cmy),(0,69,255),4)####red
        cv2.line(frame, (lpx, lpy), (cpx, cpy), (255,0, 0),4)###blue
        cv2.line(frame, (lrx, lry), (crx, cry), (0, 255, 255),4)###yellow

        cv2.imshow("kalman",frame)
        cv2.waitKey(1)

        env.goal = [goal.position.x_val, goal.position.y_val, 0]
        #env.goal = [current_prediction[0][0], current_prediction[1][0], 0]
        if distance > 0.1:

            a = get_action(o,True)
            if discrete:
                a=a[0]


            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            #img = o[0]
            #img = np.hstack(img)
            #img = np.array(img, dtype=np.uint8)
            # img = np.array(obs[0][0] , dtype=np.uint8)
            #cv2.imshow("0", img)
            #cv2.waitKey(1)


            o = o2

def det(model,device,name,goal_share):

    client = airsim.MultirotorClient('127.0.0.1')
    client.confirmConnection()
    gs = GridSpec(1, 1)
    while True:

        batch = []
        imgs = []

        for cam_id in [name]:
            responses = client.simGetImages([airsim.ImageRequest(cam_id, airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            if ((responses[0].width != 0 or responses[0].height != 0)):
                img_rgba = img1d.reshape((response.height, response.width, 4))
                rgb = Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB))  ##rgb
                imgs.append(rgb)
                batch.append(img_process(rgb))

        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], 0)
        targets = transposed_batch[1]

        images = images.to(device)

        with torch.no_grad():

            output = model(images, targets)

            if isinstance(output, torch.Tensor):
                output = [output]

        # '''
        fig = plt.figure(figsize=(3.00, 3.00), dpi=100)

        for i, prediction in enumerate(output):
            if len(prediction) == 0:

                ax = fig.add_subplot(gs[0, i])
                ax.imshow(imgs[i])
            else:
                res = []
                for p in prediction:
                    p = p.cpu().numpy()
                    p = p.round(4)
                    type = ID_TYPE_CONVERSION[int(p[0])]
                    row = [type, 0, 0] + p[1:].tolist()
                    res.append(row)

                P2 = sim_mat
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)

                ax = fig.add_subplot(gs[0, i])

                # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
                image = imgs[i]

                for line_p in res:
                    truncated = np.abs(float(line_p[1]))
                    trunc_level = 255

                    # truncated object in dataset is not observable
                    if line_p[0] in VEHICLES and truncated < trunc_level:
                        if line_p[0] == 'Car':
                            continue
                            color = 'green'
                        if line_p[0] == 'Cyclist':
                            color = 'cyan'
                        elif line_p[0] == 'Pedestrian':
                            color = 'cyan'
                        draw_3Dbox(ax, P2, line_p, color)

                ax.imshow(image)
                for line_p in res:
                    if line_p[0] == 'Pedestrian':
                        if name == "3d":
                            goal_share[0] = line_p[13]*0.7
                            goal_share[1] = line_p[11]*0.7

                        if name == "3d-back":
                            goal_share[2] = -line_p[13]*0.7
                            goal_share[3] = -line_p[11]*0.7

                        if name == "3d-right":
                            goal_share[4] = -line_p[11]*0.7
                            goal_share[5] = line_p[13]*0.7

                        if name == "3d-left":
                            goal_share[6] = line_p[11]*0.7
                            goal_share[7] = -line_p[13]*0.7
                        break
                    if line_p[0] == 'Cyclist':
                        if name == "3d":
                            goal_share[0] = line_p[13]*0.7
                            goal_share[1] = line_p[11]*0.7

                        if name == "3d-back":
                            goal_share[2] = -line_p[13]*0.7
                            goal_share[3] = -line_p[11]*0.7

                        if name == "3d-right":
                            goal_share[4] = -line_p[11]*0.7
                            goal_share[5] = line_p[13]*0.7

                        if name == "3d-left":
                            goal_share[6] = line_p[11]*0.7
                            goal_share[7] = -line_p[13]*0.7
                        break

            fig.canvas.draw()

            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close("all")
            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow("plot" + name, img)

            cv2.waitKey(1)
            # '''




def initialize_model(device):

    model_dir = path('./results') / 'AirSimEnv-v42' / 'SAC' / 'run6' / 'models'
    ac = torch.load(str(model_dir) + "/actor_model_309000" + ".pt")['model']
    ac.to(device)
    print(ac)

    cpu_device = torch.device("cpu")
    cfg = setup(args)
    model = build_detection_model(cfg)

    model.to(device)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    model.eval()

    print("successful to load detection model!")

    return ac,model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument("--config-file", default="/home/winter/zjiang/SMOKE/configs/smoke_gn_vector.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    #torch.set_num_threads(torch.get_num_threads())
    ac,model=initialize_model(device)

    model_1=deepcopy(model)
    model_2 = deepcopy(model)
    model_3 = deepcopy(model)
    mp.set_start_method('forkserver', force=True)
    ac.share_memory()
    model.share_memory()
    model_1.share_memory()
    model_2.share_memory()
    model_3.share_memory()
    goal_share=Array("d",[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    ##use pipe to asyn the detected goal

    process = [mp.Process(target=det, args=(model, device,"3d",goal_share)),
               mp.Process(target=det, args=(model_1, device,"3d-back",goal_share)),
               mp.Process(target=det, args=(model_2, device,"3d-left",goal_share)),
               mp.Process(target=det, args=(model_3, device,"3d-right",goal_share)),
               mp.Process(target=sac, args=(ac, device, goal_share)),]
    [p.start() for p in process]
    [p.join() for p in process]
