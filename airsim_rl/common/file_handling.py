import os
from os import listdir
from os.path import isfile, join
from distutils.dir_util import copy_tree
import platform
import time
from settings_folder import settings
import msgs
import re



# find list of files within a directory
def find_list_of_files(data_dir):
	assert (os.path.isdir(data_dir)), data_dir + " dir can't be found"
	file_list = []
	for f in listdir(data_dir):
		file_path = join(data_dir, f)
		if isfile(file_path) and not ("~" in f) and not (".gitignore" == f) and not ("meta_data" in f) and not (
				"test_res" in f) and not ("actor" in f) and not (".ckpt" in f) and not ("checkpoint" in f)\
				and not (".pb" in f):
			file_list.append(file_path)
	return file_list


# gets a file name and if there is a digit in it, it'll increment it and return a new filename
def incr_file_name(file_path):
	file_name = os.path.basename(file_path)
	file_name_pieces = re.split('_|\.', file_name)
	number = list(filter(lambda x: x.isdigit(), file_name_pieces))[0]
	new_file_name = file_name.replace(number, str((int(number) + 1)))
	return join(os.path.dirname(file_path), new_file_name)


def get_time_stamp(path_to_file):
	if platform.system() == 'Windows':
		return time.ctime(os.path.getmtime(path_to_file))
	else:
		stat = os.stat(path_to_file)
		try:
			return stat.st_birthtime
		except AttributeError:
			# We're probably on Linux. No easy way to get creation dates here,
			# so we'll settle for when its content was last modified.
			return stat.st_mtime


def take_first(item):
	return item[0]


def strip_test_number(item):
	res = item.split("\\")[-1].split(".")[1].split("_")[1][4:]
	return int(res)

def sort_files_based_on_name(file_list, mode):
	# sort(reverse = (mode == 'reverse'))
	return sorted(file_list, key=strip_test_number, reverse=(mode == 'newest'))


def sort_files(file_list, mode):
	file_time_stamp_dic = list(map(lambda x: (get_time_stamp(x), x), file_list))
	# sort(reverse = (mode == 'reverse'))
	return (list(map(lambda x: x[1], sorted(file_time_stamp_dic, key=take_first, reverse=(mode == 'newest')))))


def find_file_or_dir(file_list, mode):
	file_time_stamp_dic = list(map(lambda x: (get_time_stamp(x), x), file_list))
	# sort(reverse = (mode == 'reverse'))
	return (sorted(file_time_stamp_dic, key=take_first, reverse=(mode == 'newest')))[0][1]


def find_unempty_dirs(top_data_dir):
	data_dir_name = listdir(top_data_dir)
	data_dir_filtered_name = list(filter(lambda dir: dir in settings.list_algo, data_dir_name))
	data_dirs_path = [join(top_data_dir, dir) for dir in data_dir_filtered_name]
	unempty_dirs = []
	for dir in data_dirs_path:
		for parent, dirnames, filenames in os.walk(dir):
			file_list = list(filter(lambda x: not (".gitignore" in x), filenames))
			if len(file_list) > 0:
				unempty_dirs.append(dir)
				break
	return unempty_dirs


def find_all_weight_files(algo, root_dir):
	algo_data_dir_path = os.path.join(root_dir, "data", algo)
	check_point_files = []
	dir_files = [(parent, filenames) for parent, dirnames, filenames in os.walk(algo_data_dir_path, topdown=False)]
	for parent_file_names in dir_files:
		file_names = parent_file_names[1]
		dir_name = parent_file_names[0]
		if algo == "DQN":
			matches = list(filter(lambda x: re.search('[0-9]+\.hf5$', x), file_names))
		elif algo == "DDPG":
			matches = list(filter(lambda x: re.search('[0-9]+\.hf5_actor$', x), file_names))
		else:
			raise Exception("algo" + algo + "not defined")

		if algo == "DQN":
			check_point_files += list(map(lambda x: os.path.join(algo_data_dir_path, dir_name, x), matches))
		elif algo == "DDPG":
			check_point_files += list(map(lambda x: \
				                              {"actor": join(algo_data_dir_path, dir_name, x), \
				                               "critic": join(algo_data_dir_path, dir_name,
				                                              x.replace("actor", "critic"))},
			                              matches))
		else:
			raise Exception("algo" + algo + "not defined")
	return check_point_files


class Backup():
	def __init__(self):
		self.top_data_dir = join(settings.proj_root_path, "data")
		self.bu_dir = settings.bu_dir

	def get_backup(self):
		dirs_copied = self.copy_prev_expr_data()
		for dir in dirs_copied:
			self.cleanup_prev_expr_data(dir)

	# copy the previous experiment data
	def copy_prev_expr_data(self):
		unbackedup_dirs = find_unempty_dirs(self.top_data_dir)
		for dir in unbackedup_dirs:
			dirs_to_copy_to = self.make_dir_to_copy_to(dir)
			self.copy_dir(dir, dirs_to_copy_to)
		return unbackedup_dirs

	def cleanup_prev_expr_data(self, folder_to_cleanup):
		for parent, dirnames, filenames in os.walk(folder_to_cleanup):
			for fn in list(filter(lambda x: not (x == ".gitignore"), filenames)):
				os.remove(os.path.join(parent, fn))

	def make_dir_to_copy_to(self, dir_path):
		if "DQN" in dir_path:
			dir_to_look_at = join(self.bu_dir, "DQN")
		elif "DDPG" in dir_path:
			dir_to_look_at = join(self.bu_dir, "DDPG")

		if not (os.path.isdir(dir_to_look_at)):
			os.mkdir(dir_to_look_at)

		all_backedup_dirs_path = [join(dir_to_look_at, dir) for dir in listdir(dir_to_look_at)]
		if (len(all_backedup_dirs_path) == 0):
			dir_to_copy_to = join(dir_to_look_at, settings.backup_folder_name_style)
		else:
			dir_to_copy_to = incr_file_name(find_file_or_dir(all_backedup_dirs_path, 'newest'))

		os.mkdir(dir_to_copy_to)
		return dir_to_copy_to

	def copy_dir(self, src_folder, dest_folder):
		copy_tree(src_folder, dest_folder)


class CheckPoint():
	def __init__(self):
		self.max_chck_pt_per_zone = settings.max_chck_pt_per_zone
		self.ctr = 0

	def make_up_file_name(self, file_path_list, dir_path):
		if len(file_path_list) == 0:
			return join(dir_path, settings.chk_p_name_style_baselines)

		if len(file_path_list) < self.max_chck_pt_per_zone:
			return incr_file_name(find_file_or_dir(file_path_list, 'newest'))
		else:
			return (find_file_or_dir(file_path_list, 'oldest'))

	# import settings

	def find_file_to_check_point(self, zone_number):
		data_dir = join(settings.proj_root_path, "data", msgs.algo, "zone" + str(zone_number))
		file_path_list = find_list_of_files(data_dir)
		new_file_path = self.make_up_file_name(file_path_list, data_dir)
		return new_file_path

	def get_data_path(self):
		return str(join(settings.proj_root_path, "data", msgs.algo))

def find_meta_data_files_in_time_order(data_dir):
	assert (os.path.isdir(data_dir)), data_dir + " dir can't be found"
	file_list = []
	for f in listdir(data_dir):
		file_path = join(data_dir, f)
		if isfile(file_path) and ("meta_data" in f) and ("test" in f) :
			file_list.append(file_path)
	#file_list_ordered = sort_files(file_list, "oldest")
	file_list_ordered = sort_files_based_on_name(file_list, "oldest")
	return file_list_ordered
