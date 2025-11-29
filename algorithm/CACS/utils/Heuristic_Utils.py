# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/4/17 17:08
# File : Heuristic_Utils.py
import numpy as np

from algorithm.General_Utils import get_alliance_capacity_and_resource
from algorithm.Solution import Solution


def cal_heuristic_1(task_list: list, task_num: int):
    """
    计算 任务-任务 阶段的启发式信息

    Returns:
        (np.array)任务-任务阶段启发式信息矩阵

    """
    heuristic_1 = np.zeros(shape=(task_num + 1, task_num))
    t_list_xy = [[t[2], t[3]] for t in task_list]
    t_list_xy.insert(0, [0, 0])
    t_list_xy = np.array(t_list_xy)
    for i in range(task_num + 1):
        t_distance = np.sqrt(np.power(np.tile(t_list_xy[i], (t_list_xy.shape[0], 1)) - t_list_xy, 2).sum(axis=1))
        t_distance = t_distance[1:]
        non_zero_mask = np.where(t_distance != 0)
        non_zero_elements = t_distance[non_zero_mask]
        t_distance[non_zero_mask] = 1 / non_zero_elements
        heuristic_1[i] = t_distance

    return heuristic_1


def cal_heuristic_2_task_robot(cur_alliance_robot_list: list, pop_i: int, solution: Solution,
                               task: list, task_max_duration: float, task_capacity: np.ndarray, task_resource: np.ndarray,
                               robot_list: list, robot_type_list: list, weight_vector=None):
    """
    计算新解构造过程中第二阶段 任务-机器人 的启发式信息

    Args:
        cur_alliance_robot_list:当前联盟的机器人列表
        pop_i:种群编号
        solution:当前解
        task:当前任务
        task_max_duration:当前任务的最大执行时间
        task_capacity:当前任务需要的能力列表
        task_resource:当前任务需要的资源列表
        robot_list:机器人列表
        robot_type_list:机器人类型列表

    Returns:
        (np.array) 当前所有可选机器人的启发式信息

    """
    heuristic_2 = np.zeros(solution.robot_n)
    c_num = len(task_capacity)
    r_num = len(task_resource)
    for r in robot_list:
        if r in cur_alliance_robot_list:
            continue
        tmp_alliance_robot_list = cur_alliance_robot_list + [r]
        alliance_capacity, _ = get_alliance_capacity_and_resource(tmp_alliance_robot_list, robot_type_list, c_num, r_num)
        alliance_capacity[alliance_capacity == 0] = 0.0001
        a_cost = np.max(task_capacity / alliance_capacity) * task_max_duration  # 取最大用时作为成本
        tmp_alliance = [0, a_cost, tmp_alliance_robot_list]

        task_finish_time, all_travel_time, task_start_time = \
            solution.cpt_one_task_alliance_objs(task, tmp_alliance, robot_type_list)
        objs_increment = solution.cpt_objs_increment(task_finish_time, all_travel_time, task_start_time)
        if pop_i == -1:
            assert weight_vector is not None
            heuristic_2[r[0]] = 1 / (np.dot(weight_vector, (objs_increment / (objs_increment + solution.objs))) + 1)
        else:
            heuristic_2[r[0]] = 1 / ((objs_increment[pop_i] / (objs_increment[pop_i] + solution.objs[pop_i])) + 1)
    return heuristic_2
