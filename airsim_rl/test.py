# 首先，导入库文件（包括gym模块和gym中的渲染模块）
import gym
from gym.envs.classic_control import rendering


# 我们生成一个类，该类继承 gym.Env. 同时，可以添加元数据，改变渲染环境时的参数
class Test(gym.Env):
    # 如果你不想改参数，下面可以不用写
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 我们在初始函数中定义一个 viewer ，即画板
    def __init__(self):
        self.viewer = rendering.Viewer(800, 800)  # 600x400 是画板的长和框

    # 继承Env render函数
    def render(self, mode='human', close=False):
        # 画一个直径为 30 的园
        goal = rendering.make_circle(30)
        goal.set_color(0.25,0.78,0.1)
        # 添加一个平移操作

        circle_transform = rendering.Transform(translation=(100, 200))
        # 让圆添加平移这个属性,
        goal.add_attr(circle_transform)

        self.viewer.add_geom(goal)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
# 最后运行
if __name__ == '__main__':
    import airsim
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    client = airsim.MultirotorClient('127.0.0.1')
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    client.moveByVelocityZAsync(0, 0, 0, 1).join()
    while True:
        pos=client.simGetObjectPose("person")
        print(pos.position.x_val-86,
              pos.position.y_val + 13,
              pos.position.z_val -2)
        responses = client.simGetImages([airsim.ImageRequest("3d", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape((response.height, response.width, 4))
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            #cv2.imshow("0", rgb)
            #cv2.waitKey(1)

