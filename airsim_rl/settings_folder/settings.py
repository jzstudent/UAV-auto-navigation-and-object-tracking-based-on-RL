import os
import settings_folder.machine_dependent_settings as mds

# ---------------------------
# imports 
# ---------------------------
# Augmenting the sys.path with relavant folders
settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path + "/..")
os.sys.path.insert(0, proj_root_path)

# used for game configuration handling
# change it to adapt to your computer
json_file_addr = mds.json_file_addr

# used for start, restart and killing the game
game_file = mds.game_file
unreal_host_shared_dir = mds.unreal_host_shared_dir
unreal_exec = mds.unreal_exe_path

# ---------------------------
# range
# ---------------------------

easy_range_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [5,10],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[27, 27, 10],[30,30,10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0, 2]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(8, 12))}
medium_range_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [5,10],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[40, 40, 10],[45, 45, 10],[55, 55, 10],[50, 50, 10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0, 4]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(15, 22))}
hard_range_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [5,10],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[65, 65, 10],[55, 55, 10],[60, 60, 10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0, 5]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(25, 35))}
default_range_dic = easy_range_dic
# ------------------------------------------------------------
#-game related-
# ------------------------------------------------------------
game_proc_pid = ''  # process associa

# ---------------------------
# sampling frequency
# ---------------------------
end_randomization_mode = "inclusive"  # whether each level of difficulty should be inclusive (including the previous level) or exclusive

# how frequently to update the environment this is based on epides
#哪些变量可以在重启unreal后可以改变，以及改变的周期
environment_change_frequency = {"ArenaSize":5, "Seed": 5, "NumberOfObjects": 5, "End": 3, "Walls1": 3,"MinimumDistance":3}

# ------------------------------------------------------------
#                               -Drone related-
## ------------------------------------------------------------
#ip = '10.243.49.243'
ip = '127.0.0.1'

# ---------------------------
# parameters
# ---------------------------
vel_duration=0.4
double_dqn = False
mv_fw_dur = 0.4
mv_fw_spd_1 = 1* 0.5
mv_fw_spd_2 = 2* 0.5
mv_fw_spd_3 = 3* 0.5
mv_fw_spd_4 = 4* 0.5
mv_fw_spd_5 = 5* 0.5
rot_dur = 0.1#0.15
# yaw_rate = (180/180)*math.pi #in degree
yaw_rate_1_1 = 108.  # FOV of front camera
yaw_rate_1_2 = yaw_rate_1_1 * 0.5  # yaw right by this angle
yaw_rate_1_4 = yaw_rate_1_2 * 0.5
yaw_rate_1_8 = yaw_rate_1_4 * 0.5
yaw_rate_1_16 = yaw_rate_1_8 * 0.5
yaw_rate_2_1 = -108.  # -2 time the FOV of front camera
yaw_rate_2_2 = yaw_rate_2_1 * 0.5  # yaw left by this angle
yaw_rate_2_4 = yaw_rate_2_2 * 0.5
yaw_rate_2_8 = yaw_rate_2_4 * 0.5
yaw_rate_2_16 = yaw_rate_2_8 * 0.5


# ---------------------------
# general params
# ---------------------------
nb_max_episodes_steps = 512*3  # pay attention
success_distance_to_goal = 2
slow_down_activation_distance =  2*success_distance_to_goal  # detrmines at which distance we will punish the higher velocities

# ---------------------------
# reseting params
# ---------------------------
connection_count_threshold = 20  # the upper bound to try to connect to multirouter
restart_game_from_scratch_count_threshold = 3  # the upper bound to try to reload unreal from scratch
window_restart_ctr_threshold = 2  # how many times we are allowed to restart the window
# before easying up the randomization

#-------------------------------
#control mode
control_mode="Discrete" # "moveByVelocity" "Discrete"

#-------------------------------
#algorithm
algo = 'PPO'
