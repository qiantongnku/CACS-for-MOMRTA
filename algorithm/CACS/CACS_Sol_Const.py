# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/5/19 15:20
# File : CACS_Sol_Const.py
import pickle

import numpy as np

from algorithm.General_Utils import classify_task_by_pre, wheel_select, alliance_robots_to_alliance_id
from algorithm.CACS.utils.Heuristic_Utils import cal_heuristic_2_task_robot
from algorithm.Solution import Solution


def generate_new_solution(task_list, task_type_list, robot_list, robot_type_list, weight_vectors,
                          pheromone_1, pheromone_2, heuristic_1, alpha, beta, select_r, pop_i):
    """
    产生新解
    """
    robot_num = len(robot_list)
    task_num = len(task_list)
    new_solution = Solution(robot_num, task_num, weight_vectors[pop_i])
    new_task_sequence = []
    select_phi_1 = (pheromone_1[pop_i] ** alpha) * (heuristic_1 ** beta)  # 任务-任务 阶段选择概率矩阵
    parent_t, child_t = classify_task_by_pre(pickle.loads(pickle.dumps(task_list)))
    while len(parent_t) != 0:

        # 1) 选择任务
        if len(new_task_sequence) == 0:  # 选择第一个任务
            next_task_phi = select_phi_1[0][parent_t]
        else:  # 选择其他任务
            next_task_phi = select_phi_1[new_task_sequence[-1] + 1][parent_t]
        next_task_id = wheel_select(next_task_phi, parent_t, select_r)  # 任务选择
        next_task = task_list[next_task_id]  # 选出的下一个任务
        task_max_duration = task_type_list[next_task[1]][1]  # 任务最大执行时间
        task_capacity = np.array(task_type_list[next_task[1]][2])  # 任务需要的能力列表
        task_resource = np.array(task_type_list[next_task[1]][3])  # 任务需要的资源列表

        # 2) 选择联盟
        next_alliance = []
        alliance_robots_list = []
        alliance_capacity = np.zeros_like(task_capacity)  # 联盟提供的能力列表
        alliance_resource = np.zeros_like(task_resource)  # 联盟提供的资源列表

        flag = False
        while not flag:
            # 计算当前联盟和每个可选机器人的启发式信息
            heuristic_2 = cal_heuristic_2_task_robot(alliance_robots_list, pop_i, new_solution, next_task,
                                                     task_max_duration, task_capacity, task_resource,
                                                     robot_list, robot_type_list)
            select_phi_2 = pheromone_2[pop_i][next_task[0]] * (heuristic_2 ** beta)
            # 一步步选择机器人组成联盟
            robot = wheel_select(select_phi_2, robot_list, select_r)
            alliance_robots_list.append(robot)
            alliance_capacity += np.array(robot_type_list[robot[1]][2])
            alliance_resource += np.array(robot_type_list[robot[1]][3])

            # 判断当前联盟能否执行任务
            if (alliance_capacity >= task_capacity).all() and (alliance_resource >= task_resource).all():
                alliance_id = alliance_robots_to_alliance_id(alliance_robots_list, robot_num)
                alliance_cost = np.max(task_capacity[alliance_capacity != 0] /
                                       alliance_capacity[alliance_capacity != 0]) * task_max_duration  # 取最大用时作为成本
                next_alliance = [alliance_id, alliance_cost, alliance_robots_list]
                flag = True
            else:
                select_phi_2[robot[0]] = 0

        new_solution.add_one_task_alliance(next_task, next_alliance, robot_type_list)
        new_task_sequence.append(next_task_id)

        # 3) 更新可选择的任务集合
        for t in child_t:
            if next_task_id in t[1]:
                t[1].remove(next_task_id)
                if len(t[1]) == 0:
                    parent_t.append(t[0])
        parent_t.remove(next_task_id)

    new_solution.type = pop_i + 1

    return new_solution
