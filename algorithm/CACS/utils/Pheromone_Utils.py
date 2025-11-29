# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/4/17 17:07
# File : Pheromone_Utils.py
import pickle
import random

import numpy as np

from algorithm.General_Utils import classify_task_by_pre, alliance_id_to_alliance_robots, select_one_alliance
from algorithm.Solution import Solution


def init_random_solution(task_list: list, task_type_list: list, robot_list: list, robot_type_list: list,
                         robot_num: int, task_num: int, weight_vector: np.ndarray):
    """
    生成随机初始解

    Returns:
        (Solution) 随机策略生成的初始解

    """
    s = Solution(robot_num, task_num, weight_vector)
    task_list_copy = pickle.loads(pickle.dumps(task_list))
    parent_t, child_t = classify_task_by_pre(task_list_copy)
    while len(parent_t) != 0:
        # 1) 每次随机选取一个不带有先序约束的任务
        task_id = random.choice(parent_t)
        task = task_list[task_id]
        # 2) 使用随机策略选择一个联盟
        alliance = select_one_alliance(task, task_type_list, robot_list, robot_type_list, robot_num, select_type=1)
        s.add_one_task_alliance(task, alliance, robot_type_list)
        # 3) 将执行完毕的任务从先序约束中剔除
        for t in child_t:
            if task_id in t[1]:
                t[1].remove(task_id)
                if len(t[1]) == 0:
                    parent_t.append(t[0])
        parent_t.remove(task_id)
    return s


def init_pheromone(pop_num: int, task_list: list, task_type_list: list, robot_list: list, robot_type_list: list,
                   robot_num: int, task_num: int):
    """
    初始化信息素矩阵

    Returns:
        (np.array)各种群对应的权重向量;
        (np.array)信息素初始值;
        (np.array)任务-任务阶段信息素矩阵;
        (np.array)任务-机器人阶段信息素矩阵;

    """
    weight_vectors = np.zeros(shape=(pop_num, pop_num))
    pheromone_0 = np.zeros(pop_num)
    pheromone_1 = np.zeros(shape=(pop_num, task_num + 1, task_num))
    pheromone_2 = np.zeros(shape=(pop_num, task_num, robot_num))
    for pop_i in range(pop_num):
        # 1) 完全随机策略生成初始解
        weight_vectors[pop_i][pop_i] += 1
        s = init_random_solution(task_list, task_type_list, robot_list, robot_type_list, robot_num, task_num,
                                 weight_vectors[pop_i])
        # 2) 计算信息素初始值
        pheromone_0[pop_i] = 1 / (task_num * s.F)
        # 3) 初始化 任务-任务 任务-联盟 阶段信息素矩阵
        pheromone_1[pop_i] += pheromone_0[pop_i]
        pheromone_2[pop_i] += pheromone_0[pop_i]
    return weight_vectors, pheromone_0, pheromone_1, pheromone_2


def local_update_pheromone(s: Solution, lose_rate: float, pheromone_0: float, pheromone_1: np.ndarray,
                           pheromone_2: np.ndarray, robot_list: list):
    """
    信息素局部更新

    Args:
        s:更新解
        lose_rate:蒸发率
        pheromone_0:信息素初始值
        pheromone_1:任务-任务阶段信息素矩阵
        pheromone_2:任务-联盟阶段信息素矩阵
        robot_list:机器人列表

    Returns:
        (np.array) 任务-任务 阶段信息素矩阵; (dict) 任务-机器人 阶段信息素矩阵

    """
    result_1 = pheromone_1
    result_2 = pheromone_2
    # 1) 信息素蒸发
    # 2) 新的信息素累积
    increment = lose_rate * pheromone_0
    result_1[0][s.t_a_s[0][0]] = result_1[0][s.t_a_s[0][0]] * (1 - lose_rate) + increment
    for i in range(s.task_n - 1):
        cur_task = s.t_a_s[i][0]
        cur_alliance = s.t_a_s[i][1]
        next_task = s.t_a_s[i + 1][0]
        result_1[cur_task + 1][next_task] = result_1[cur_task + 1][next_task] * (1 - lose_rate) + increment
        alliance_robots = alliance_id_to_alliance_robots(cur_alliance, robot_list)
        for r in alliance_robots:
            result_2[cur_task][r[0]] = result_2[cur_task][r[0]] * (1 - lose_rate) + increment
    alliance_robots = alliance_id_to_alliance_robots(s.t_a_s[-1][1], robot_list)
    for r in alliance_robots:
        result_2[s.t_a_s[-1][0]][r[0]] = result_2[s.t_a_s[-1][0]][r[0]] * (1 - lose_rate) + increment

    return result_1, result_2


def global_update_pheromone(pop_num: int, archive: list, global_rho: float, pheromone_1: np.ndarray,
                            pheromone_2: np.ndarray, robot_list: list):
    """
    信息素全局更新: 对每个单目标种群, 从Archive中排序选择一个最优解进行全局更新

    Args:
        pop_num:种群数
        archive:非支配解集
        global_rho:信息素全局更新蒸发率
        pheromone_1:任务-任务 信息素矩阵
        pheromone_2:任务-机器人 信息素矩阵
        robot_list:机器人列表

    Returns:
        pheromone_1, pheromone_2

    """
    # 1) 信息素蒸发 (全局)
    result_1 = pheromone_1 * (1 - global_rho)
    result_2 = pheromone_2 * (1 - global_rho)

    for p in range(pop_num):
        # 对Archive解进行排序, 选取对于种群p最优的解进行全局更新
        tmp_objs = [s.objs[p] for s in archive]
        tmp_objs = np.array(tmp_objs)
        global_s = archive[np.argmin(tmp_objs)]
        global_increment = global_rho / global_s.objs[p]

        # 2) 新的信息素累积
        result_1[p][0][global_s.t_a_s[0][0]] += global_increment
        for i in range(global_s.task_n - 1):
            cur_task = global_s.t_a_s[i][0]
            cur_alliance = global_s.t_a_s[i][1]
            next_task = global_s.t_a_s[i + 1][0]
            result_1[p][cur_task + 1][next_task] += global_increment
            alliance_robots = alliance_id_to_alliance_robots(cur_alliance, robot_list)
            for r in alliance_robots:
                result_2[p][cur_task][r[0]] += global_increment
        alliance_robots = alliance_id_to_alliance_robots(global_s.t_a_s[-1][1], robot_list)
        for r in alliance_robots:
            result_2[p][global_s.t_a_s[-1][0]][r[0]] += global_increment

    return result_1, result_2

