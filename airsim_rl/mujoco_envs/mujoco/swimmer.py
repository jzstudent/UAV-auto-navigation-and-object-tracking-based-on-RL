import gym
import numpy as np
from gym.envs.mujoco import SwimmerEnv as SwimmerEnv_

class SwimmerEnv(SwimmerEnv_):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer().render()


class SwimmerVelEnv(SwimmerEnv):
    def __init__(self, task={}, low=0.0, high=2.0):
        self._task = task
        self.low = low
        self.high = high

        self._goal_vel = task.get('velocity', 0.0)
        super(SwimmerEnv, self).__init__()

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']


    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = -np.abs((xposafter - xposbefore) / self.dt - self._goal_vel)
        reward_ctrl = ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd - reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)
