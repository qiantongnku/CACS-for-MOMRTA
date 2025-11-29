# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/4/17 16:34
# File : GA_Utils.py
import random
import numpy as np
import pickle

from algorithm.General_Utils import is_alliance_available, alliance_id_to_alliance_robots
from algorithm.Solution import Solution


def cross(p1: Solution, p2: Solution, task_list: list):
    """
    交叉算子: 对两个父代解的任务序列进行三段式交叉

    Args:
        p1:父代解1
        p2:父代解2
        task_list:任务列表

    Returns:
        (Solution) 交叉产生的新解

    """
    task_n = p1.task_n
    a = np.array(p1.t_a_s)
    b = np.array(p2.t_a_s)
    pos_start = random.randint(0, task_n-2)
    pos_end = random.randint(pos_start+1, task_n-1)
    a1 = a[:pos_start]
    a2 = a[pos_start:pos_end]
    a3 = a[pos_end:]
    b1 = b[:pos_start]
    b2 = b[pos_start:pos_end]
    b3 = b[pos_end:]
    new_a = generate_one_crossover_tas(a1, a2, a3, b2, b, task_list)
    new_b = generate_one_crossover_tas(b1, b2, b3, a2, a, task_list)

    return new_a, new_b


def generate_one_crossover_tas(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, b2: np.ndarray, b: np.ndarray,
                               task_list: list):
    """
    任务序列交叉: 给定父代解的片段,按照a3a1b2的顺序生成新的子代解

    Returns:
        np.array 交叉得到的新解

    """
    # 1) 去除重复的任务
    a3a1 = np.concatenate((a3, a1))
    delete_index = []
    for t in b2[:, 0]:
        if t not in a2[:, 0]:
            delete_index.append(np.where(a3a1[:, 0] == t)[0][0])
    a3a1 = np.delete(a3a1, delete_index, axis=0)

    # 2) 填补缺少的任务
    task_to_add = np.union1d(a2[:, 0], b2[:, 0])
    new_b2 = np.zeros(shape=(len(task_to_add), 3), dtype=int)
    for i in range(len(task_to_add)):
        t = task_to_add[i]
        # 获取在b中的索引
        t_index = np.where(b[:, 0] == t)[0]
        assert len(t_index) == 1
        new_b2[i][0] = t_index[0]
        new_b2[i][1] = t
        new_b2[i][2] = b[t_index[0]][1]
    new_b2_sort = new_b2[np.argsort(new_b2[:, 0])]
    new_a = np.concatenate((a3a1, new_b2_sort[:, [1, 2]]))

    # 3) 修复先序约束
    flag = True
    count = 0
    while flag:
        count +=1
        flag = False
        visited_count = len(new_a) - 1
        for i in range(len(new_a) - 1, -1, -1):
            t = task_list[new_a[i][0]]
            if len(t[4]) != 0:
                switch_index = []
                visited_task = new_a[visited_count:, 0]
                for j in range(len(visited_task)):
                    if visited_task[j] in t[4]:
                        switch_index.append(i + j)
                # 与最后一个先序任务交换位置
                if len(switch_index) != 0:
                    k = max(switch_index)
                    tmp = new_a[i:i + 1, :].copy()
                    new_a[i:i + 1, :] = new_a[k:k + 1, :]
                    new_a[k:k + 1, :] = tmp
                    flag = True
            visited_count -= 1

    return new_a


def mutate(origin_t_a_s: np.ndarray, task_list: list, task_type_list: list, robot_list: list, robot_type_list: list,
           robot_num: int, rate: float):
    """
    变异算子: 对解的每个机器人联盟根据概率进行突变, 只保留可行变异

    Args:
        origin_t_a_s:原解的 任务-联盟序列
        task_list:任务列表
        task_type_list:任务类型列表
        robot_list:机器人列表
        robot_type_list:机器人类型列表
        robot_num:机器人数量
        rate:每个联盟变异的概率

    Returns:
        (list) 返回变异后的任务-联盟序列

    """
    t_a_s = pickle.loads(pickle.dumps(origin_t_a_s))
    for j in range(len(t_a_s)):
        r = random.uniform(0, 1)
        if r < rate:
            cur_task_id = t_a_s[j][0]
            cur_alliance_id = t_a_s[j][1]
            new_alliance_id = mutate_available_alliance(cur_task_id, cur_alliance_id, task_list, task_type_list,
                                                        robot_list, robot_type_list, robot_num)
            if new_alliance_id == -1:
                new_alliance_id = cur_alliance_id
            t_a_s[j][1] = new_alliance_id
    return t_a_s


def mutate_available_alliance(cur_task_id: int, cur_alliance_id: int, task_list: list, task_type_list: list,
                              robot_list: list, robot_type_list: list, robot_num: int):
    """
    二进制变异产生一个新的联盟

    Args:
        cur_task_id:当前的任务id
        cur_alliance_id:当前的联盟id
        task_list:任务列表
        task_type_list:任务类型列表
        robot_list:机器人列表
        robot_type_list:机器人类型列表
        robot_num:机器人数量

    Returns:
        (int) 突变产生的新的联盟id

    """
    cur_task = task_list[cur_task_id]
    cur_alliance_id_01 = bin(cur_alliance_id).replace("0b", "")
    while len(cur_alliance_id_01) < robot_num:
        cur_alliance_id_01 = '0' + cur_alliance_id_01
    new_alliance_id = -1
    for _ in range(robot_num):
        # 随机选取一位进行二进制取反操作
        i = random.randint(0, robot_num - 1)
        bit = '1' if cur_alliance_id_01[i] == '0' else '0'
        new_alliance_id_01 = cur_alliance_id_01[:i] + bit + cur_alliance_id_01[i + 1:]
        # 将变异后的二进制联盟索引转换为十进制
        tmp_alliance_id = int(new_alliance_id_01, 2)
        new_alliance_robot_list = alliance_id_to_alliance_robots(tmp_alliance_id, robot_list)
        # 判断联盟是否可行
        if is_alliance_available(cur_task, task_type_list, new_alliance_robot_list, robot_type_list):
            # 可行,则保留变异
            new_alliance_id = tmp_alliance_id
            break
    if new_alliance_id == -1:
        new_alliance_id = cur_alliance_id

    return new_alliance_id
