# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/3/16 20:58
# File : General_Utils.py
import logging
import random

import numpy as np


def classify_task_by_pre(task_list: list):
    """
    将任务根据是否带有先序约束进行划分

    Args:
        task_list:任务列表

    Returns:
        (list)不带先序约束的父任务列表;(list)带有先序约束的子任务列表

    """
    parent_t = []
    child_t = []
    for t in task_list:
        if not t[4]:
            parent_t.append(t[0])
        else:
            child_t.append([t[0], t[4]])
    return parent_t, child_t


def cal_dominate_count(solution_set: list):
    """
    计算解集中每个解被支配的解数量

    Args:
        solution_set:解种群

    Returns:
        (np.array) 表示解集中每个解被支配的解的数量

    """
    dominate_count = np.zeros(len(solution_set))
    for i in range(len(solution_set) - 1):
        s_i = solution_set[i]
        for j in range(i + 1, len(solution_set)):
            s_j = solution_set[j]
            if s_i.is_dominate(s_j):
                dominate_count[j] += 1
            elif s_j.is_dominate(s_i):
                dominate_count[i] += 1
    return dominate_count


def update_archive(source: list, target: list):
    """
    将当前解集合中的非支配解更新到目标解集合中

    Args:
        source:当前解集合
        target:目解标集合

    Returns:
        (np.array) source加入target集合的非支配解的索引

    """
    if len(source) == 0:
        return np.array([])
    # 1) 挑选种群中的非支配解加入到target中
    pop_dominate_count = cal_dominate_count(source)
    non_dominate_index = np.argwhere(pop_dominate_count == 0)
    if len(non_dominate_index) != 0:
        non_dominate_index = non_dominate_index.reshape(-1)
        for nd_i in non_dominate_index:
            nd_s = source[nd_i]
            flag = True
            for ep_s in target:
                if nd_s.t_a_s == ep_s.t_a_s:
                    flag = False
                    break
            if flag:
                target.append(nd_s)
        # 2) 加入新解后, 移除target中被支配解
        ep_dominate_count = cal_dominate_count(target)
        ep_dominate_index = np.argwhere(ep_dominate_count > 0)
        if len(ep_dominate_index) != 0:
            ep_dominate_index = ep_dominate_index.reshape(-1)
            for i in range(len(ep_dominate_index) - 1, -1, -1):
                target.pop(ep_dominate_index[i])
    return non_dominate_index


def wheel_select(select_phi, select_elements, select_R):
    """
    轮盘赌选择元素

    Args:
        select_phi:选择概率
        select_elements:要选择的元素列表
        select_R:选择比例参数

    Returns:
        选择出的元素索引

    """
    r = random.uniform(0, 1)

    if type(select_phi) == dict:  # 阶段2的选择概率是dict类型
        phi = np.array(list(select_phi.values()))
        elements = np.array(list(select_phi.keys()))
    else:
        phi = select_phi
        elements = select_elements

    if r < select_R:  # 直接选取最大概率的元素
        next_element = elements[np.argmax(phi)]
    else:  # 执行轮盘赌
        assert len(phi) == len(elements)
        indexes = np.arange(len(elements))
        # if len(phi) == 1 and phi[0] == 0:
        #     phi[0] = 1
        if phi.sum() == 0:
            index = random.choice(indexes)
            print('Something goes wrong...')
        else:
            index = np.random.choice(indexes, size=1, p=phi / phi.sum())[0]
        next_element = elements[index]
    return next_element


# ######################## #
#            联盟           #
# ######################## #
def select_one_alliance(task: list, task_type_list: list, robot_list: list, robot_type_list: list, robot_num: int,
                        select_type: int, select_phi: np.ndarray = None, select_r: int = None):
    """
    选择一个可行联盟: 1) 随机选择 2) 根据选择概率进行轮盘赌

    Args:
        task:当前任务
        task_type_list:任务类型列表
        robot_list:机器人列表
        robot_type_list:机器人类型列表
        robot_num:机器人数量
        select_type:选择类型
        select_phi:选择概率
        select_r:选择控制参数

    Returns:
        (list) alliance
    """
    task_type = task[1]
    task_max_duration = task_type_list[task_type][1]  # 任务最大完成时间
    task_capacity = np.array(task_type_list[task_type][2])  # 任务需要的能力列表
    task_resource = np.array(task_type_list[task_type][3])  # 任务需要的资源列表
    alliance = []
    alliance_robot_list = []
    alliance_capacity = np.zeros_like(task_capacity)  # 联盟提供的能力列表
    alliance_resource = np.zeros_like(task_resource)  # 联盟提供的资源列表

    robot_list_copy = [r for r in robot_list]
    flag = False
    while not flag:
        if select_type == 1:  # 随机选择
            robot = random.choice(robot_list_copy)  # 每次随机选择一个机器人加入到联盟中
        else:  # 轮盘赌选择
            assert select_phi is not None and select_r is not None
            robot = wheel_select(select_phi, robot_list_copy, select_r)
        alliance_robot_list.append(robot)
        alliance_capacity += np.array(robot_type_list[robot[1]][2])
        alliance_resource += np.array(robot_type_list[robot[1]][3])

        # 判断联盟能否执行任务
        if (alliance_capacity >= task_capacity).all() and (alliance_resource >= task_resource).all():
            alliance_id = alliance_robots_to_alliance_id(alliance_robot_list, robot_num)
            alliance_cost = np.max(task_capacity[alliance_capacity != 0] /
                                   alliance_capacity[alliance_capacity != 0]) * task_max_duration  # 取最大用时作为成本
            alliance = [alliance_id, alliance_cost, alliance_robot_list]
            flag = True
        else:
            if select_type == 1:
                robot_list_copy.remove(robot)
            else:
                select_phi[robot[0]] = 0

    return alliance


def get_alliance_by_id(alliance_id, task_type, robot_list, robot_type_list):
    task_max_duration = task_type[1]  # 任务最大完成时间
    task_capacity = np.array(task_type[2])  # 任务需要的能力列表
    task_resource = np.array(task_type[3])  # 任务需要的资源列表
    alliance_robot_list = alliance_id_to_alliance_robots(alliance_id, robot_list)
    alliance_capacity, alliance_resource = get_alliance_capacity_and_resource(alliance_robot_list, robot_type_list,
                                                                              len(task_capacity), len(task_resource))
    alliance_cost = np.max(task_capacity[alliance_capacity != 0] /
                           alliance_capacity[alliance_capacity != 0]) * task_max_duration  # 取最大用时作为成本
    alliance = [alliance_id, alliance_cost, alliance_robot_list]
    return alliance


def get_alliance_capacity_and_resource(alliance_robot_list: list, robot_type_list: list, capacity_num: int,
                                       resource_num: int):
    """
    计算联盟提供的能力和资源列表

    Args:
        alliance_robot_list:联盟中的机器人列表, [[r_id, rt_id], ...]
        robot_type_list:机器人类型列表
        capacity_num:能力数量
        resource_num:资源数量

    Returns:
        (np.array)联盟提供的能力列表, (np.array)联盟提供的资源列表

    """
    alliance_capacity = np.zeros(capacity_num)  # 联盟提供的能力列表
    alliance_resource = np.zeros(resource_num)  # 联盟提供的资源列表
    for r in alliance_robot_list:
        alliance_capacity += np.array(robot_type_list[r[1]][2])
        alliance_resource += np.array(robot_type_list[r[1]][3])
    return alliance_capacity, alliance_resource


def is_alliance_available(task, task_type_list, alliance_robot_list, robot_type_list):
    task_type = task[1]
    task_capacity = np.array(task_type_list[task_type][2])  # 任务需要的能力列表
    task_resource = np.array(task_type_list[task_type][3])  # 任务需要的资源列表
    alliance_capacity = np.zeros_like(task_capacity)  # 联盟提供的能力列表
    alliance_resource = np.zeros_like(task_resource)  # 联盟提供的资源列表
    for r in alliance_robot_list:
        alliance_capacity += np.array(robot_type_list[r[1]][2])
        alliance_resource += np.array(robot_type_list[r[1]][3])
    if (alliance_capacity >= task_capacity).all() and (alliance_resource >= task_resource).all():
        return True
    else:
        return False


def alliance_id_to_alliance_robots(alliance_index: int, robot_list: list):
    """
    根据联盟的十进制索引，获取联盟中的成员（通过二进制转换）

    Args:
        alliance_index: 联盟在成本矩阵中的索引
        robot_list: 所有的机器人列表

    Returns:
        (list[robot]) 联盟包含的机器人列表

    """
    alliance_robots = []
    # 索引转化为对应的2进制数
    alliance_index_01 = bin(alliance_index).replace("0b", "")
    while len(alliance_index_01) < len(robot_list):
        alliance_index_01 = '0' + alliance_index_01
    for i in range(len(alliance_index_01)):
        # 判断包含哪些机器人
        if alliance_index_01[len(alliance_index_01) - i - 1] == '1':
            # print("选取第 %d 个机器人" % (i + 1))
            alliance_robots.append(robot_list[i])
    return alliance_robots


def alliance_robots_to_alliance_id(alliance_robot_list: list, robot_num: int):
    """
    联盟机器人列表转换为十进制联盟id

    Args:
        alliance_robot_list:联盟包含的机器人列表
        robot_num:机器人总数

    Returns:
        联盟十进制id

    """
    binary_id = ''
    for _ in range(robot_num):
        binary_id += '0'
    for robot in alliance_robot_list:
        binary_id = binary_id[:robot_num - 1 - robot[0]] + '1' + binary_id[len(binary_id) - robot[0]:]
    return int(binary_id, 2)


def log_eval(name: str,
             max_eval_num: int,
             instance_name: str,
             seed_num: int,
             start_time: float,
             end_time: float,
             start_timestamp: float,
             end_timestamp: float,
             start_actual_time: str,
             end_actual_time: str,
             count: int,
             archive):
    # 配置日志文件
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('./log/' + name + '_' + instance_name + '.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 写入日志文件
    logger.info(instance_name + ' - Seed: ' + str(seed_num))
    logger.info("Start Actual Time:" + start_actual_time)
    logger.info('Start CPU Time: ' + str(start_time) + 's')
    logger.info("End Actual Time:" + end_actual_time)
    logger.info('End CPU Time: ' + str(end_time) + 's')
    logger.info("Total Actual Time: {:.2f}s".format(end_timestamp - start_timestamp))  # 打印程序运行时间
    logger.info("Total CPU Time: " + str(end_time - start_time) + "s")
    logger.info("Eval: " + str(max_eval_num) + " Iter: " + str(count) + "  Archive Size: " + str(len(archive)))
    logger.info("=================================================================================")
    logger.removeHandler(file_handler)
