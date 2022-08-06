import airsim
import numpy as np
import math
import time
import cv2
from settings_folder import settings



class AirLearningClient(object):
    def __init__(self):

        self.last_img = np.zeros((1, 112, 112))
        self.last_grey = np.zeros((112, 112))
        self.last_rgb = np.zeros((112, 112, 3))
        self.width, self.height=84,84 ##deepmind settings

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient(settings.ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        #self.z=-3
        self.z = -0.9

    def goal_direction(self, goal, pos):

        pitch, roll, yaw = self.client.getPitchRollYaw()
        yaw = math.degrees(yaw)

        pos_angle = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)
        #return ((math.degrees(track) - 180) % 360) - 180

        return np.array([track])

    def getScreenRGB(self):
        responses = self.client.simGetImage("3d", airsim.ImageType.Scene)
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape((response.height, response.width, 4))
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            self.last_rgb=rgb
        else:
            print("Something bad happened! Restting AirSim!")
            #self.AirSim_reset()

            rgb=self.last_rgb
        #rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return rgb

    def getScreenDepth(self):
        responses = self.client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True,
                                                                  False)
                                              ]
                                             ,vehicle_name='multirotor')
        #responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis,True, False)])

        if (responses == None):
            print("Camera is not returning image!")
            print("Image size:" + str(responses[0].height) + "," + str(responses[0].width))
            img = [np.array([0]) for _ in responses]
        else:
            img = []
            for res in responses:
                img.append(np.array(res.image_data_float, dtype=np.float))
            img = np.stack(img, axis=0)


        ##pre-process for depth img
        img = img.clip(max=20)
        scale=255/20
        img=img*scale#, dtype=np.uint8)

        img2d=[]
        for i in range(len(responses)):
            if ((responses[i].width != 0 or responses[i].height != 0)):
                img2d.append(np.reshape(img[i], (responses[i].height, responses[i].width)))
            else:
                print("Something bad happened! Restting AirSim!")
                img2d.append(self.last_img[i])

        self.last_img = np.stack(img2d, axis=0)

        if len(img2d)>1:
            return img2d
        else:
            return img2d[0]

    def get_ryp(self):
        pitch, roll, yaw = self.client.getPitchRollYaw()
        return np.array([yaw])

    def drone_pos(self):
        x = self.client.getPosition().x_val
        y = self.client.getPosition().y_val
        z = self.client.getPosition().z_val

        return np.array([x, y, z])

    def drone_velocity(self):
        v_x = self.client.getVelocity().x_val
        v_y = self.client.getVelocity().y_val
        v_z = self.client.getVelocity().z_val
        speed = np.sqrt(v_x ** 2 + v_y ** 2)
        return np.array([v_x, v_y, speed])

    def get_distance(self, goal):
        now = self.client.getPosition()
        xdistance = (goal[0] - now.x_val)
        ydistance = (goal[1] - now.y_val)
        #zdistance = (goal[2] - now.z_val)
        euclidean = np.sqrt(np.power(xdistance,2) + np.power(ydistance,2))
        return np.array([xdistance, ydistance])

    def get_velocity(self):
        return np.array([self.client.getVelocity().x_val, self.client.getVelocity().y_val, self.client.getVelocity().z_val])

    def AirSim_reset(self):
        self.client=airsim.MultirotorClient(settings.ip)
        connection_established = False
        # wait till connected to the multi rotor
        while not (connection_established):
            try:
                time.sleep(1)
                connection_established = self.client.confirmConnection()
            except Exception as e:
                #self.client.reset()
                time.sleep(5)
                self.client = airsim.MultirotorClient(settings.ip)


        #self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def unreal_reset(self):
        #!!!这个时间间隔很重要！！！！
        self.client.resetUnreal(1.5,2.5)###1,2.5


    def take_continious_action(self, action):

        if(settings.control_mode=="moveByVelocity"):
            action=np.clip(action, -0.3, 0.3)

            detla_x = action[0]
            detla_y = action[1]
            v=self.drone_velocity()
            v_x = v[0] + detla_x
            v_y = v[1] + detla_y

            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, 0.35, 1, yaw_mode).join()

        else:
            raise NotImplementedError

        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)

        return collided
        #Todo : Stabilize drone


    def straight(self, speed, duration):
        pitch, roll, yaw  = self.client.getPitchRollYaw()
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 1, yaw_mode).join()


    def move_right(self, speed, duration):
        pitch, roll, yaw = self.client.getPitchRollYaw()
        vx = math.sin(yaw) * speed
        vy = math.cos(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 0).join()
        start = time.time()
        return start, duration

    def yaw_right(self, rate, duration):
        self.client.rotateByYawRateAsync(rate, duration).join()
        start = time.time()
        return start, duration

    def pitch_up(self, duration):
        self.client.moveByVelocityAsync(0,0,1,duration,1).join()
        start = time.time()
        return start, duration

    def pitch_down(self, duration):
        #yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityAsync(0,0,-1,duration,1).join()
        start = time.time()
        return start, duration

    def move_forward_Speed(self, speed_x = 0.5, speed_y = 0.5, duration = 0.5):
        #speedx is in the FLU
        #z = self.drone_pos()[2]
        pitch, roll, yaw  = self.client.getPitchRollYaw()
        vel = self.client.getVelocity()
        vx = math.cos(yaw) * speed_x + math.sin(yaw) * speed_y
        vy = math.sin(yaw) * speed_x - math.cos(yaw) * speed_y

        drivetrain = 1
        yaw_mode = airsim.YawMode(is_rate= False, yaw_or_rate = 0)

        self.client.moveByVelocityZAsync(vx = (vx +vel.x_val)/2 ,
                             vy = (vy +vel.y_val)/2 , #do this to try and smooth the movement
                             z = self.z,
                             duration = duration,
                             drivetrain = drivetrain,
                             yaw_mode=yaw_mode
                            ).join()
        start = time.time()
        return start, duration

    def take_discrete_action(self, action):

        if action == 0:
            self.straight(settings.mv_fw_spd_2, settings.rot_dur)
        if action == 1:
            self.straight(settings.mv_fw_spd_3, settings.rot_dur)
        if action == 2:
            #self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur/2)
            #self.straight(settings.mv_fw_spd_3, settings.rot_dur/2)
            self.move_forward_Speed(settings.mv_fw_spd_2*math.cos(0.314),
                                    settings.mv_fw_spd_2*math.sin(0.314), settings.rot_dur)

        if action == 3:
            #self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_3 * math.cos(0.314),
                                    settings.mv_fw_spd_3 * math.sin(0.314), settings.rot_dur)

        if action == 4:
            #self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_2 * math.cos(0.314),
                                    -settings.mv_fw_spd_2 * math.sin(0.314), settings.rot_dur)
        if action == 5:
            #self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_3 * math.cos(0.314),
                                    -settings.mv_fw_spd_3 * math.sin(0.314), settings.rot_dur)

        if action == 6:
            self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur )
        if action == 7:
            self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        '''
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        if action == 0:
            v = self.drone_velocity()
            v_x = v[0] + 0.25
            v_y = v[1] + 0
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 1:
            v = self.drone_velocity()
            v_x = v[0] - 0.25
            v_y = v[1] + 0
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 2:
            v = self.drone_velocity()
            v_x = v[0] + 0
            v_y = v[1] + 0.25
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 3:
            v = self.drone_velocity()
            v_x = v[0] + 0
            v_y = v[1] - 0.25
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()


        
        if action == 0:
            start, duration = self.straight(settings.mv_fw_spd_4, settings.mv_fw_dur)
        if action == 1:
            start, duration = self.straight(settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 2:
            start, duration = self.straight(settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 3:
            start, duration = self.straight(settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 4:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 5:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 6:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, settings.mv_fw_spd_1, settings.mv_fw_dur)
        if action == 7:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, -settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 8:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, -settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 9:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, -settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 10:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_4, settings.mv_fw_dur)
        if action == 11:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 12:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 13:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 14:
            start, duration = self.yaw_right(settings.yaw_rate_1_1, settings.rot_dur)
        if action == 15:
            start, duration = self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur)
        if action == 16:
            start, duration = self.yaw_right(settings.yaw_rate_1_4, settings.rot_dur)
        if action == 17:
            start, duration = self.yaw_right(settings.yaw_rate_1_8, settings.rot_dur)

        if action == 18:
            start, duration = self.yaw_right(settings.yaw_rate_2_1, settings.rot_dur)
        if action == 19:
            start, duration = self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        if action == 20:
            start, duration = self.yaw_right(settings.yaw_rate_2_4, settings.rot_dur)
        if action == 21:
            start, duration = self.yaw_right(settings.yaw_rate_2_8, settings.rot_dur)
        '''

        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)

        return collided

