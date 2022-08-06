#!/usr/bin/env python
import airsim
import math
import time
import numpy as np
import cv2

if __name__ == '__main__':
    client = airsim.MultirotorClient('127.0.0.1')
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    airsim.wait_key('Press any key to takeoff')
    # print(client.getMultirotorState())
    # print(airsim.to_eularian_angles(client.simGetVehiclePose().orientation))
    f1 = client.takeoffAsync().join()
    h = -0.1
    s = 0.5
    # f1=client.moveToPositionAsync(0, 0, h, 0.5).join()
    # f1=client.hoverAsync(vehicle_name="drone_1").join()
    print("##########OK!##########\n")
    yaw = 0
    while (1):
        control = input("press WASD to control dron_1!")
        print("you press %s\n" % control)
        state = client.getMultirotorState().kinematics_estimated.position
        responses = client.simGetImages([airsim.ImageRequest("mycamera", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape((response.height, response.width, 4))
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)  ###别怀疑这里确实是rgb图像
            cv2.imwrite("aaa.png", rgb)
        x = state.x_val
        y = state.y_val
        z = state.z_val
        print("your position is (%s,%s,%s)\n" % (x, y, z))
        if control == "d":
            yaw += 10
            print("you want yaw to be:%s" % yaw)
            client.moveByVelocityAsync(0, 0, 0, 1, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                       airsim.YawMode(False, yaw)).join()
        elif control == "a":
            yaw -= 10
            print("you want yaw to be:%s" % yaw)
            client.moveByVelocityAsync(0, 0, 0, 1, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                       airsim.YawMode(False, yaw)).join()
        elif control == "w":
            client.moveByVelocityAsync(s * math.cos(yaw * math.pi / 180), s * math.sin(yaw * math.pi / 180), 0, 1,
                                       airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, yaw)).join()
        elif control == "s":
            client.moveByVelocityAsync(-s * math.cos(yaw * math.pi / 180), -s * math.sin(yaw * math.pi / 180), 0, 1,
                                       airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, yaw)).join()
        elif control == "q":
            client.hoverAsync(vehicle_name="drone_1").join()
            break
    yaw = 0




