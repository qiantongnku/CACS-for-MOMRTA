# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/5/19 14:41
# File : CACS_Local_Search.py
import random

import numpy as np
from algorithm.CACS.utils.GA_Utils import cross, mutate
from algorithm.Solution import Solution


def local_search_eval(total_evaluate_num, max_evaluate_num, single_pop_size, archive, crossover_rate, mutation_rate,
                      task_list, task_type_list, robot_list, robot_type_list, robot_num, task_num, pop_num):
    """
    以最大解评估次数为终止条件的局部搜索

    Args:
        total_evaluate_num:
        max_evaluate_num:
        single_pop_size:
        archive:
        crossover_rate:
        mutation_rate:
        task_list:
        task_type_list:
        robot_list:
        robot_type_list:
        robot_num:
        task_num:
        pop_num:

    Returns:
        (list[Solution]) 局部搜索得到的新的种群; (int) 新的评估次数

    """
    new_pop = []
    if len(archive) < 2:
        return new_pop, total_evaluate_num
    for _ in range(single_pop_size):
        # 1) 选择
        [p1, p2] = random.sample(archive, 2)
        # 2) 交叉
        cro_r = random.uniform(0, 1)
        if cro_r < crossover_rate:
            cro_a, cro_b = cross(p1, p2, task_list)  # 得到交叉后的t_a_list
        else:
            cro_a = p1.t_a_s
            cro_b = p2.t_a_s
        # 3) 变异
        mut_r = random.uniform(0, 1)
        if mut_r < mutation_rate:
            mut_a = mutate(cro_a, task_list, task_type_list, robot_list, robot_type_list, robot_num, 0.5)
            mut_b = mutate(cro_b, task_list, task_type_list, robot_list, robot_type_list, robot_num, 0.5)
        else:
            mut_a = cro_a
            mut_b = cro_b

        # 4) 新解加入新种群
        a = Solution(robot_num, task_num, np.zeros(pop_num))
        a.set_new_tas(mut_a, task_list, task_type_list, robot_list, robot_type_list)
        a.type = 4
        new_pop.append(a)
        total_evaluate_num += 1
        if total_evaluate_num >= max_evaluate_num:
            break
        b = Solution(robot_num, task_num, np.zeros(pop_num))
        b.set_new_tas(mut_b, task_list, task_type_list, robot_list, robot_type_list)
        b.type = 4
        new_pop.append(b)
        total_evaluate_num += 1
        if total_evaluate_num >= max_evaluate_num:
            break
    return new_pop, total_evaluate_num


def cal_crowding_distance_by_Manhattan(archive_objs_01: np.ndarray):
    """
    用曼哈顿距离计算Archive所有解的拥挤度

    Args:
        archive_objs_01:

    Returns:

    """
    archive_num = len(archive_objs_01)
    result = np.zeros(archive_num)
    for i in range(archive_num):
        result[i] = np.sum(np.abs(archive_objs_01 - archive_objs_01[i]))
    return result


def crossover_two_point(a1: np.ndarray,
                        a2: np.ndarray,
                        a3: np.ndarray,
                        b2: np.ndarray,
                        b: np.ndarray,
                        task_list: list):
    """
    任务序列两点交叉: 给定父代解的片段,按照a1b2a3的顺序生成新的子代解

    Returns:
        np.array 交叉得到的新解的任务联盟序列

    """
    # 1) 去除重复的任务
    a1_tasks = a1[:, 0]
    a3_tasks = a3[:, 0]
    a1_delete_index = []
    a3_delete_index = []
    for [task_id, _] in b2:
        if task_id in a1_tasks:
            a1_delete_index.append(np.where(a1_tasks == task_id)[0][0])
        elif task_id in a3_tasks:
            a3_delete_index.append(np.where(a3_tasks == task_id)[0][0])
    new_a1 = np.delete(a1, a1_delete_index, axis=0)
    new_a3 = np.delete(a3, a3_delete_index, axis=0)
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
    new_a = np.concatenate((new_a1, new_b2_sort[:, [1, 2]], new_a3))
    # 3) 修复先序约束
    flag = True
    count = 0
    while flag:
        count += 1
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