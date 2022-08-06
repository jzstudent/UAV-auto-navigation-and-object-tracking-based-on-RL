from settings_folder import settings
import msgs
from gym.envs.classic_control import rendering
from environment_randomization.game_config_handler_class import *
import gym
import collections
from gym import spaces
from gym.utils import seeding
from gym_airsim.envs.airlearningclient import *
from common.utils import *


class AirSimEnv(gym.Env):
    def __init__(self, need_render=False):

        # if need_render is True, then we can use the 2d windows to render the env

        STATE_RGB_H, STATE_RGB_W = 112,112

        self.stack_frames=4

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.stack_frames, STATE_RGB_H, STATE_RGB_W))


        #eithor your speed is more than 2, or your duration is more than 0.4
        #otherwise, it dose not work well !
        if (settings.control_mode == "moveByVelocity"):
            self.action_space = spaces.Box(np.array([-0.3, -0.3]),
                                           np.array([+0.3, +0.3]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(8)

        #UE4 env config
        self.game_config_handler = GameConfigHandler()

        #uav api
        self.airgym = AirLearningClient()

        #reset the env var
        self.success_count = 0
        self.episodeN = 0
        self.stepN = 0
        self.goal = airsimize_coordinates(self.game_config_handler.get_cur_item("End"))


        self.prev_state = self.init_state_f()
        self.init_state = self.prev_state
        self.success = False
        self.level=0
        self.success_deque = collections.deque(maxlen=100)
        self.seed()

        self.need_render=need_render
        if self.need_render:
            self.viewer = rendering.Viewer(1000, 1000)

    def getGoal(self):
        return self.goal

    def get_space(self):
        return self.observation_space,self.action_space

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def print_msg_of_inspiration(self):
        if (self.success_count %2 == 0):
            print("---------------:) :) :) Success, Be Happy (: (: (:------------ !!!\n")
        elif (self.success_count %3 == 0):
            print("---------------:) :) :) Success, Shake Your Butt (: (: (:------------ !!!\n")
        else:
            print("---------------:) :) :) Success, Oh Yeah! (: (: (:------------ !!!\n")

    def init_state_f(self):
        now = self.airgym.drone_pos()[:2]
        pry = self.airgym.get_ryp()
        d=[]
        for i in range(self.stack_frames):
            d.append(self.airgym.getScreenDepth())
            time.sleep(0.03)
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)
        self.velocity = self.airgym.drone_velocity()
        self.speed = self.velocity[2]
        inform=np.concatenate((self.relative_position, self.velocity, pry, self.r_yaw))
        d=np.stack(d)
        return [d, inform]

    def state(self):
        now = self.airgym.drone_pos()[:2]
        pry = self.airgym.get_ryp()
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)
        self.velocity = self.airgym.drone_velocity()
        self.speed = self.velocity[2]
        inform=np.concatenate((self.relative_position, self.velocity, pry, self.r_yaw))
        return inform

    def computeReward(self, now):

        distance_now = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                               + np.power((self.goal[1] - now[1]), 2)
                               )
        now = self.airgym.drone_pos()[:2]
        r_yaw = self.airgym.goal_direction(self.goal, now)

        r = -distance_now*0.03#0.02

        if math.cos(r_yaw)>=0:
            r += self.speed*math.cos(r_yaw)

        return r


    #根据控制飞机飞的方式不同，动作空间和step会有不同
    #根据不同输入，obs space会有不同
    def step(self, action):

        self.stepN += 1
        action = action[0]

        #self.airgym.client.simPause(False)
        if (settings.control_mode == "moveByVelocity"):

            collided = self.airgym.take_continious_action(action)

        else:
            collided = self.airgym.take_discrete_action(action)

        #self.airgym.client.simPause(True)

        #update state
        inform = self.state()

        d=self.airgym.getScreenDepth()
        state = []
        for i in range(self.stack_frames):
            if i <(self.stack_frames-1):
                state.append(self.prev_state[0][i+1])
            else:
                state.append(d)
        state=[state, inform]
        now = self.airgym.drone_pos()

        print("ENter Step" + str(self.stepN))
        print("Relative Position:" + str(self.relative_position))
        print("Success count:",self.success_count)
        #print("Speed:"+str(self.speed))
        print("Action:",action)
        print("Goal:" + str(self.goal))

        distance = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                           +np.power((self.goal[1] - now[1]), 2)
                           )

        if distance < settings.success_distance_to_goal:
            self.success_count += 1
            done = True
            self.print_msg_of_inspiration()
            self.success = True
            msgs.success = True
            reward = 20.0

        elif collided == True:
            done = True
            reward = -20.0
            self.success = False

        elif self.stepN >= settings.nb_max_episodes_steps:
            done = True
            reward = -20.0
            self.success = False
        elif (now[2] < -4.5) or (now[2]>-0.7):  # Penalize for flying away too high or too low
            done = True
            reward = -20.0
            self.success = False

        else:
            reward = self.computeReward(now)
            done = False
            self.success = False

        # Todo: penalize for more crazy and unstable actions
        print("rew:", reward)

        self.prev_state = state


        if (done):
            if self.success:
                self.success_deque.append(1)
            else:
                self.success_deque.append(0)
            self.on_episode_end()

        return state, reward, done, None


    def on_episode_end(self):
        pass


    def on_episode_start(self):
        self.stepN = 0
        self.episodeN += 1

    def reset(self):

        #'''
        if len(self.success_deque)>0:
            succes_rate=sum(self.success_deque) / len(self.success_deque)
            if succes_rate>0.7 and self.level==0 and self.success_count>300:
                self.level=1
                self.game_config_handler=GameConfigHandler(range_dic_name="settings.medium_range_dic")
            elif succes_rate > 0.7 and self.level == 1 and self.success_count>600:
                self.level = 2
                self.game_config_handler = GameConfigHandler(range_dic_name="settings.hard_range_dic")
        #'''
        if self.need_render:
            self.viewer.geoms.clear()
            self.viewer.onetime_geoms.clear()
        print("enter reset")
        self.randomize_env()
        print("done randomizing")
        self.airgym.unreal_reset()
        print("done unreal_resetting")
        time.sleep(4)# if your pc is a rubbish then make it sleep long
        self.airgym.AirSim_reset()
        print("done arisim reseting")

        self.airgym.client.takeoffAsync().join()

        now = self.airgym.drone_pos()

        ##sometimes there may occur something you can't imagine! Just like your uav is dancing~

        while (-now[2])<0.5:
            ####TODO:change env

            vars_to_randomize = ['Seed']
            self.sampleGameConfig(*vars_to_randomize)
            self.airgym.unreal_reset()
            print("done unreal_resetting")
            time.sleep(4)
            self.airgym.AirSim_reset()
            self.airgym.client.takeoffAsync().join()
            now = self.airgym.drone_pos()

        self.airgym.client.moveByVelocityZAsync(0,0,self.airgym.z, 1).join()

        self.goal=np.array([30,40,0])


        self.on_episode_start()
        print("done on episode start")

        state = self.init_state_f()
        self.prev_state = state

        return state

    def randomize_env(self):
        vars_to_randomize = []
        for k, v in settings.environment_change_frequency.items():
            if (self.episodeN+1) %  v == 0:
                vars_to_randomize.append(k)

        if (len(vars_to_randomize) > 0):
            self.sampleGameConfig(*vars_to_randomize)
            self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))


    def updateJson(self, *args):
        self.game_config_handler.update_json(*args)

    def getItemCurGameConfig(self, key):
        return self.game_config_handler.get_cur_item(key)

    def setRangeGameConfig(self, *args):
        self.game_config_handler.set_range(*args)

    def getRangeGameConfig(self, key):
        return self.game_config_handler.get_range(key)

    def sampleGameConfig(self, *arg):
        self.game_config_handler.sample(*arg, np_random=self.np_random)

    def render(self, mode='human', close=False):
        # 画一个直径为 30 的circle
        #self.viewer.geoms.clear()
        #self.viewer.onetime_geoms.clear()

        goal = rendering.make_circle(30)
        goal.set_color(0.25, 0.78, 0.1)
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(self.goal[0]*10+500, self.goal[1]*10+500))
        # 让圆添加平移这个属性
        goal.add_attr(circle_transform)
        self.viewer.add_geom(goal)

        uav = rendering.make_circle(10)
        uav.set_color(0.78, 0.37, 0.66)
        now = self.airgym.drone_pos()
        uav_transform = rendering.Transform(translation=(now[0]*10 + 500, now[1]*10 + +500))
        uav.add_attr(uav_transform)
        self.viewer.add_geom(uav)

        size = self.game_config_handler.get_cur_item("ArenaSize")
        h = size[0] * 10
        w = size[1] * 10

        line1 = rendering.Line(((1000 - h) / 2, (1000 - w) / 2), ((h + 1000) / 2, (1000 - w) / 2))
        line2 = rendering.Line(((1000 - h) / 2, (1000 - w) / 2), ((1000 - h) / 2, (w + 1000) / 2))
        line3 = rendering.Line(((h + 1000) / 2, (1000 - w) / 2), ((h + 1000) / 2, (1000 + w) / 2))
        line4 = rendering.Line(((1000 - h) / 2, (1000 + w) / 2), ((h + 1000) / 2, (1000 + w) / 2))
        # 给元素添加颜色
        line1.set_color(0, 0, 0)
        line2.set_color(0, 0, 0)
        line3.set_color(0, 0, 0)
        line4.set_color(0, 0, 0)
        self.viewer.add_geom(line1)
        self.viewer.add_geom(line2)
        self.viewer.add_geom(line3)
        self.viewer.add_geom(line4)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')