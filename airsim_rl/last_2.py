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


import torch
from torchvision.transforms import functional as Func
import os
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.patches as patches
from matplotlib.path import Path

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
sim_mat=[245.52978,0.000000,224.581720,0.000000,
        0.000000,245.65646,117.289441,0.000000,
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

    K = np.array(sim_mat, dtype=np.float32).reshape(3, 4)
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


def sac(device, seed=100, total_steps=int(5000), replay_size=int(1e5), gamma=0.99,
        polyak=0.995, lr=5e-4, alpha=0.2, batch_size=256, start_steps=5000,
        update_after=10000, update_every=50, save_freq=3000, sattn= True):
    cpu_device = torch.device("cpu")
    cfg = setup(args)
    model = build_detection_model(cfg)
    #device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    model.eval()

    print("successful to load detection model!")


    client = airsim.MultirotorClient('127.0.0.1')
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveByVelocityZAsync(0, 0, -1, 1).join()

    # Main loop: collect experience in env and update/log each epoch
    for t in trange(total_steps):

        batch = []
        imgs=[]

        start = time.time()
        responses = \
            client.simGetImages([airsim.ImageRequest("3d", airsim.ImageType.Scene, False, False)])

        print(time.time() - start)
        start = time.time()
        for i in range(1):
            response = responses[i]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            if ((response.width != 0 or response.height != 0)):
                img_rgba = img1d.reshape((response.height, response.width, 4))
                rgb = Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB))  ##rgb
                imgs.append(rgb)
                batch.append(img_process(rgb))
        print(time.time() - start)
        start = time.time()
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], 0)
        targets = transposed_batch[1]

        images = images.to(device)

        with torch.no_grad():

            output = model(images, targets)

            if isinstance(output, torch.Tensor):
                output=[output]
        print(time.time() - start)
        #'''
        fig = plt.figure(figsize=(20.00, 5.00), dpi=100)
        gs = GridSpec(1, 4)
        for i,prediction in enumerate(output):
            if len(prediction) == 0:
                print([])
                ax = fig.add_subplot(gs[0, i])
                image = imgs[i]
                ax.imshow(image)
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
                #gs.update(wspace=i)  # set the spacing between axes.

                ax = fig.add_subplot(gs[0, i])

                # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
                image = imgs[i]
                for line_p in res:
                    truncated = np.abs(float(line_p[1]))
                    trunc_level = 255

                    # truncated object in dataset is not observable
                    if line_p[0] in VEHICLES and truncated < trunc_level:
                        if line_p[0] == 'Car':
                            color = 'green'
                        if line_p[0] == 'Cyclist':
                            color = 'yellow'
                        elif line_p[0] == 'Pedestrian':
                            color = 'cyan'
                        draw_3Dbox(ax, P2, line_p, color)

                # visualize 3D bounding box
                ax.imshow(image)

        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close("all")
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow("plot", img)

        # display camera feed

        cv2.waitKey(1)
        #'''




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

    #if torch.cuda.is_available():
    #    device = torch.device("cuda:1")
    #    torch.set_num_threads(1)
    #else:
    #    device = torch.device("cpu")

    device = torch.device("cuda:0")
    torch.set_num_threads(torch.get_num_threads())

    sac(device=device)