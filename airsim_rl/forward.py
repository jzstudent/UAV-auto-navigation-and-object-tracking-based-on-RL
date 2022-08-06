from smoke.config import cfg
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from smoke.modeling.detector import build_detection_model
from smoke.utils import comm
from smoke.modeling.heatmap_coder import get_transfrom_matrix
from smoke.structures.params_3d import ParamsList
from smoke.structures.image_list import to_image_list


import torch
from torchvision.transforms import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.patches as patches
from matplotlib.path import Path


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
        return F.to_tensor(image), target


class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image, target):
        if self.to_bgr:
            image = image[[2, 1, 0]]
        image = F.normalize(image, mean=self.mean, std=self.std)
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

def img_process(idx):
    # load default parameter here
    img_path = os.path.join("./smoke_origin", idx)
    img = Image.open(img_path)
    K=tello_mat
    K = np.array(K, dtype=np.float32).reshape(3, 4)
    K = K[:3, :3]

    center = np.array([i / 2 for i in img.size], dtype=np.float32)
    size = np.array([i for i in img.size], dtype=np.float32)

    """
    resize, horizontal flip, and affine augmentation are performed here.
    since it is complicated to compute heatmap w.r.t transform.
    """

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

    images = to_image_list(img, 0)
    targets = [target]
    img_ids = [idx]
    return images,targets,img_ids



def main(args):
    cfg = setup(args)
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for idx in os.listdir("./smoke_origin"):
        images, targets, image_ids = img_process(idx)
        images = images.to(device)
        with torch.no_grad():
            output = model(images, targets)
            output = output.to(cpu_device)

        results_dict.update(
            {img_id: output for img_id in image_ids}
        )
    predictions=results_dict

    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            print([])
        else:
            res=[]
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                res.append(row)


            image_file = os.path.join("./smoke_origin", image_id)
            P2 = tello_mat
            P2 = np.array(P2, dtype=np.float32).reshape(3, 4)

            fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
            gs = GridSpec(1, 4)
            gs.update(wspace=0)  # set the spacing between axes.

            ax = fig.add_subplot(gs[0, :3])

            # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
            image = Image.open(image_file).convert('RGB')

            for line_p in res:
                truncated = np.abs(float(line_p[1]))
                trunc_level = 255

                # truncated object in dataset is not observable
                if line_p[0] in VEHICLES and truncated < trunc_level:
                    color = 'green'
                    if line_p[0] == 'Cyclist':
                        color = 'yellow'
                    elif line_p[0] == 'Pedestrian':
                        color = 'cyan'
                    draw_3Dbox(ax, P2, line_p, color)

            # visualize 3D bounding box
            ax.imshow(image)
            ax.set_xticks([])  # remove axis value
            ax.set_yticks([])

            plt.show()
            fig.savefig(os.path.join("./smoke_ret", image_id),
                        dpi=fig.dpi, bbox_inches='tight', pad_inches=0)

    comm.synchronize()



ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}
VEHICLES=['Car', 'Cyclist', 'Pedestrian']
sim_mat=[1047.52978,0.000000,958.581720,0.000000,
        0.000000,1047.65646,505.289441,0.000000,
        0.000000,0.000000,1.000000,0.000000]

tello_mat=[1891.67382,0.0000, 1239.04049,0.000000,
            0.0000000,1889.65445,8.71607713e+02,0.000000,
            0.000000,0.00000,1.00000,0.000000]


if __name__ == '__main__':
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
