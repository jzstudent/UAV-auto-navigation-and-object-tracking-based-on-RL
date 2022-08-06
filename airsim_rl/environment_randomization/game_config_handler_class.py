from settings_folder import settings
import os
from environment_randomization.game_config_class import *
import copy
from common import utils
from common.file_handling import *

class GameConfigHandler:
    def __init__(self,
                 range_dic_name="settings.default_range_dic",
                 input_file_addr=settings.json_file_addr):

        range_dic=eval(range_dic_name)
        assert (os.path.isfile(input_file_addr)), input_file_addr + " doesnt exist"
        self.input_file_addr = input_file_addr

        #当前game中环境config
        self.cur_game_config = GameConfig(input_file_addr)

        #game中元素的变化范围
        self.game_config_range = copy.deepcopy(self.cur_game_config)
        self.set_range(*[el for el in range_dic.items()])#items将dict转成元组对


    def set_items_without_modifying_json(self, *arg):
        for el in arg:
            assert (type(
                el) is tuple), el + " needs to be tuple, i.e. the input needs to be provided in the form of (key, new_value)"
            key = el[0]
            value = el[1]
            assert (key in self.cur_game_config.find_all_keys()), key + " is not a key in the json file"
            self.cur_game_config.set_item(key, value)

    def update_json(self, *arg):
        self.set_items_without_modifying_json(*arg)
        outputfile = self.input_file_addr
        output_file_handle = open(outputfile, "w")
        json.dump(self.cur_game_config.config_data, output_file_handle)
        output_file_handle.close()

    def get_cur_item(self, key):
        return self.cur_game_config.get_item(key)

    def set_range(self, *arg):
        for el in arg:

            assert (type(el) is tuple), str(
                el) + " needs to be tuple, i.e. the input needs to be provided in the form of (key, range)"
            assert (type(el[1]) is list), str(
                el) + " needs to be list, i.e. the range needs to be provided in the form of [lower_bound,..., upper_bound]"
            key = el[0]
            value = el[1]
            assert (key in self.cur_game_config.find_all_keys()), key + " is not a key in the json file"
            self.game_config_range.set_item(key, value)

    def get_range(self, key):
        return self.game_config_range.get_item(key)

    # sampling within the entire range
    def sample(self, *arg, np_random = None):
        all_keys = self.game_config_range.find_all_keys()
        if (len(arg) == 0):
            arg = all_keys

        for el in arg:
            assert (el in all_keys), str(el) + " is not a key in the json file"

            # corner cases
            if el in ["Indoor", "GameSetting"]:  # make sure to not touch indoor, cause it'll mess up the keys within it
                continue
            low_bnd = 0
            up_bnd = len(self.game_config_range.get_item(el))

            random_val = np_random.choice(list(range(low_bnd,up_bnd)))
            random_val=self.game_config_range.get_item(el)[random_val]

            self.cur_game_config.set_item(el, random_val)

        # end
        if "End" in arg and self.game_config_range.get_item("End")[0] == "Mutable":

            self.cur_game_config.set_item("End",
                                          utils.get_random_end_point(
                                          self.cur_game_config.get_item("ArenaSize"),
                                          0,
                                          1,
                                          np_random))

        outputfile = self.input_file_addr
        output_file_handle = open(outputfile, "w")
        json.dump(self.cur_game_config.config_data, output_file_handle)
        output_file_handle.close()
        if not( settings.ip == '127.0.0.1'):
            utils.copy_json_to_server(outputfile)
