# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/4/5 13:10
# File : Utils.py
import pickle


# ######################## #
#         文件处理          #
# ######################## #
def save_data(save_path, data):
    """
    把数据保存到pickle文件中

    Args:
        save_path:文件保存路径
        data:要保存的数据

    """
    with open(save_path, "wb") as file:
        pickle.dump(data, file)
    file.close()


def read_data(save_path):
    """
    从pickle文件读取数据

    Args:
        save_path:文件路径

    Returns:
        读取到的ep数据

    """
    with open(save_path, "rb") as file:
        data = pickle.load(file)
    file.close()
    return data

