import os
from os import sys
import json
import copy


class GameConfig:
    def __init__(self, input_file_addr):
        assert (os.path.isfile(input_file_addr)), input_file_addr + " doesnt exist"
        self.input_file_addr = input_file_addr  # input file to parse
        self.populate()

    def populate(self):
        with open(self.input_file_addr) as data_file:
            jstring = data_file.read().replace('nan', 'NaN')
            data = json.loads(jstring)
            self.config_data = copy.deepcopy(data)#json_file EnvGenConfig.json
            data_file.close()

    # 将所有的json中的字典的key提取出来，递归
    def find_all_keys_helper(self, in_val):

        if type(in_val) is list:
            res = map(lambda x: self.find_all_keys_helper(x), in_val)
            return sum(res, [])  # flattening the list
        if type(in_val) is dict:
            res = map(lambda x: [x] + self.find_all_keys_helper(in_val[x]), in_val)
            return sum(res, [])  # flattening the list
        return []

    def find_all_keys(self):
        return self.find_all_keys_helper(self.get())

    # 递归,把setting文件中dict的元素替换到config_data中
    def set_item_helper(self, in_val, key_to_compare, new_value):
        if type(in_val) is list:
            res = []
            for el in in_val:
                res.append(self.set_item_helper(el, key_to_compare, new_value))
        elif type(in_val) is dict:
            res = {}
            for el in in_val:
                if (el == key_to_compare):
                    res[el] = new_value
                else:
                    res[el] = self.set_item_helper(in_val[el], key_to_compare, new_value)
        else:
            res = in_val
        return res


    def set_item(self, key_to_compare, new_value):
        self.config_data = self.set_item_helper(self.get(), key_to_compare, new_value)

    def add_item(self, key, value):
        self.config_data[key] = value

    def get_item_helper(self, in_val, key_to_compare):
        res = []
        if type(in_val) is list:
            for el in in_val:
                res += self.get_item_helper(el, key_to_compare)
        elif type(in_val) is dict:
            for el in in_val:
                if (el == key_to_compare):
                    res += [in_val[el]]
                else:
                    res += self.get_item_helper(in_val[el], key_to_compare)
        return res

    def get_item(self, key_to_compare):
        value = self.get_item_helper(self.get(), key_to_compare)
        assert (len(value) > 0), key_to_compare + " doesn't exist"
        return value[0]

    def get_all_items(self):
        dic = {}
        keys = self.find_all_keys()
        for el in keys:
            dic[el] = self.get_item(el)
        return dic

    def get(self):
        return self.config_data
