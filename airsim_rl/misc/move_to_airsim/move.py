import airsim
import os

from shutil import copy2

airsim_dir = os.path.dirname(airsim.__file__)
file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(file_path)
print(airsim_dir)
copy2(os.path.join(dir_path, "client.py"), airsim_dir)
copy2(os.path.join(dir_path, "types.py"), airsim_dir)
