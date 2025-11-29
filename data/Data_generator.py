# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/2/20 21:46
# File : new_data_generator.py
import os
import random
import collections
import json

import numpy as np


def generate_random_list(num):
    """
    随机生成指定长度的float类型列表,
    并随机指定一些元素为0,
    非零元素为在[0, 1]范围内的float数

    Args:
        num:列表长度

    Returns:
        (list) 随机列表
    """
    result = [round(random.uniform(0,1),4) for _ in range(num)]
    # 指定一些元素为0
    unavailable_count = random.randint(0, num//2)
    index_list = [i for i in range(num)]
    random.shuffle(index_list)
    for i in range(unavailable_count):
        result[index_list[i]] = 0

    return result


def generate_robot_type(num_robot_type, num_capacity, num_resource, min_v, max_v):
    """
    随机生成机器人类型

    Args:
        num_robot_type:机器人类型数量
        num_capacity:该机器人类型可以提供的能力种类数量
        num_resource:该机器人类型可以提供的资源种类数量
        min_v:机器人类型的最小移动速度
        max_v:机器人类型的最大移动速度

    Returns:
        (list) 机器人类型列表, [id, v, (list)capacity_list, (list)resource_list]
    """
    robot_type_list = []
    for id in range(num_robot_type):
        # 该类型机器人可以提供的能力列表
        capacity_list = generate_random_list(num_capacity)
        # 该类型机器人可以提供的资源列表
        resource_list = generate_random_list(num_resource)
        # 该类型机器人的速度
        v = round(random.uniform(min_v, max_v),4)

        robot_type_list.append([id, v, capacity_list, resource_list])

    return robot_type_list


def generate_task_type(num_task_type, num_capacity, num_resource, min_t, max_t):
    """
    随机生成任务类型

    Args:
        num_task_type:任务类型数量
        num_capacity:执行任务需要的能力种类数
        num_resource:执行任务需要的资源种类数
        min_t:任务最大执行时间的最小值
        max_t:任务最大执行时间的最大值

    Returns:
        (list) 任务类型列表, [id, max_duration, (list)capacity_list, (list)resource_list]
    """
    task_type_list = []
    for id in range(num_task_type):
        # 该类型任务的最大执行时间
        max_duration = random.randint(min_t, max_t)
        # 执行该类型任务需要的能力列表
        capacity_list = generate_random_list(num_capacity)
        # 任务该类型执行需要的资源列表
        resource_list = generate_random_list(num_resource)

        task_type_list.append([id, max_duration, capacity_list, resource_list])

    return task_type_list


def generate_num_list(total_num, type_num):
    """
    随机生成每种类型的个体数量

    Args:
        total_num:个体总数
        type_num:类型总数

    Returns:
        (list) 每种类型的个体数量列表
    """
    num_list = []

    # 为每种类型随机生成[2,10]的个体数
    tmp_total_num = 0
    for _ in range(type_num):
        tmp = random.randint(2, 10)
        num_list.append(tmp)
        tmp_total_num += tmp

    # 检查生成的数量和是否等于个体总数
    rate = total_num / tmp_total_num
    tmp_total_num = 0
    for i in range(type_num):
        n = num_list[i]
        num_list[i] = int(n*rate)
        if num_list[i] <= 0:
            num_list[i] = 1
        tmp_total_num += num_list[i]

    # 超出/不足个体总数则进行修剪/补全
    if tmp_total_num > total_num:
        while tmp_total_num > total_num:
            max_index = 0
            for i in range(type_num):
                if num_list[i] > num_list[max_index]:
                    max_index = i
            num_list[max_index] -= 1
            tmp_total_num -= 1
    elif tmp_total_num < total_num:
        while tmp_total_num < total_num:
            min_index = 0
            for i in range(type_num):
                if num_list[i] < num_list[min_index]:
                    min_index = i
            num_list[min_index] += 1
            tmp_total_num += 1

    return num_list


def generate_robot(num_robot, num_robot_type):
    """
    随机生成机器人个体

    Args:
        num_robot:机器人数量
        num_robot_type:机器人类型数量

    Returns:
        (list) 机器人列表, [id, robot_type]
    """
    robot_list = []
    num_list = generate_num_list(num_robot, num_robot_type)
    id = 0
    while id < num_robot:
        for type in range(len(num_list)):
            for i in range(num_list[type]):
                robot_list.append([id, type])
                id += 1

    return robot_list


def generate_task(num_task, num_task_type, x_range, y_range, num_constraint):
    """
    随机生成任务

    Args:
        num_task:任务数量
        num_task_type:任务类型数量
        x_range:任务点的x坐标范围, 二元组
        y_range:任务点的y坐标范围, 二元组
        num_constraint:先序约束的数量

    Returns:
        (list) 任务列表, [id, task_type, x, y, (list)precedence_constraint]
    """
    task_list = []
    num_list = generate_num_list(num_task, num_task_type)
    id = 0
    while id < num_task:
        for type in range(len(num_list)):
            for i in range(num_list[type]):
                x = round(random.uniform(x_range[0], x_range[1]), 4)
                y = round(random.uniform(y_range[0], y_range[1]), 4)
                task_list.append([id, type, x, y, []])
                id += 1

    all_pre = generate_precedence_constraint(num_task, num_constraint)
    for pre in all_pre:
        task_list[pre[0]][4].append(int(pre[1]))

    return task_list


def generate_precedence_constraint(num_task, num_constraint):
    """
    构建随机的先序约束

    Args:
        num_task:任务数量
        num_constraint:先序约束数量

    Returns:
        (np.array) 生成的先序约束
    """
    assert (num_constraint <= num_task*(num_task-1)/2), "先序约束数量过大"

    constraints = np.zeros(shape=(num_constraint, 2), dtype=int)
    node_st = np.zeros(num_task, dtype=int)     # 记录当前任务的子任务数量
    connect = [[] for _ in range(num_task)]     # 记录当前任务的子任务

    for j in range(num_constraint):

        # 1 选择父任务
        while True:
            st = random.randint(0, num_task-2)
            if node_st[st] < (num_task-1)-st:     # 每个节点最多只能有(num_task-1)-st条边
                break

        # 2 选择子任务, 只选择比st大的任务做为子任务, 并且随着子任务数量的增加, 缩小范围
        et = random.randint(st+1, (num_task-1)-node_st[st])

        # 3 验证
        visit = []
        for i in range(len(connect[st])):
            visit.append(connect[st][i])
        visit.append(st)
        visit.sort()
        for i in range(len(visit)):
            if et < visit[i]:
                break
            elif et == visit[i]:
                et = visit[i] + 1

        node_st[st] += 1
        connect[st].append(et)
        constraints[j][0] = st
        constraints[j][1] = et

    assert not check_precedence_constraint(constraints), "先序约束存在环"

    return constraints


def check_precedence_constraint(constraints):
    """
    检查生成的先序约束中是否存在环

    Args:
        constraints:生成的先序约束列表

    Returns:
        (bool) 是否存在环
    """
    for c in constraints:
        if c[0] >= c[1]:
            return True
    return False


def check_data(robot_list, task_list, robot_type_list, task_type_list):
    """
    检查生成的实验数据是否可行
    (1) 所有机器人的总能力 > 每个类型的任务需要的能力
    (2) 所有机器人携带的总资源 > 每个类型的任务需要的资源

    Args:
        robot_list:机器人列表
        task_list:任务列表
        robot_type_list:机器人类型列表
        task_type_list:任务类型列表

    Returns:
        (bool) 数据是否满足要求
    """
    # 机器人
    for r in range(len(robot_type_list)):
        if np.sum(robot_type_list[r][2]) == 0:
            return False
        if np.sum(robot_type_list[r][3]) == 0:
            return False

    # 1 能力
    for i in range(len(task_type_list)):
        if np.sum(task_type_list[i][2]) == 0:
            return False
        for j in range(len(task_type_list[i][2])):
            task_capacity = task_type_list[i][2][j]
            robot_capacity = 0
            for r in robot_list:
                robot_capacity += robot_type_list[r[1]][2][j]

            # 所有机器人的总能力要大于每个类型的任务需要的能力
            if task_capacity > robot_capacity:
                return False

    # 2 资源
    for i in range(len(task_type_list)):
        for j in range(len(task_type_list[i][3])):
            task_resource = task_type_list[i][3][j]
            robot_resource = 0
            for r in robot_list:
                robot_resource += robot_type_list[r[1]][3][j]

            # 所有机器人的总资源要大于每个类型的任务需要的资源
            if task_resource > robot_resource:
                return False

    return True


def repair_data(robot_list, task_list, robot_type_list, task_type_list, num_capacity, num_resource):
    """
    修复不满足要求的实验数据
    (1) 所有机器人的总能力 > 每个类型的任务需要的能力
    (2) 所有机器人的总资源 > 每个类型的任务需要的资源

    Args:
        robot_list:机器人列表
        task_list:任务列表
        robot_type_list:机器人类型列表
        task_type_list:任务类型列表
        num_capacity:能力维度
        num_resource:资源维度

    Returns:
        (list) 机器人类型列表; (list) 任务类型列表
    """
    tmp = collections.Counter(np.array(robot_list)[:, 1])
    max_robot_num = max(list(tmp.values()))

    # 机器人
    for r in range(len(robot_type_list)):
        while np.sum(robot_type_list[r][2]) == 0:
            robot_type_list[r][2] = generate_random_list(num_capacity)
        while np.sum(robot_type_list[r][3]) == 0:
            robot_type_list[r][3] = generate_random_list(num_resource)


    # 1 能力
    for i in range(len(task_type_list)):
        while np.sum(task_type_list[i][2]) == 0:
            task_type_list[i][2] = generate_random_list(num_capacity)
        for j in range(len(task_type_list[i][2])):
            task_capacity = task_type_list[i][2][j]
            robot_capacity = 0
            for r in robot_list:
                robot_capacity += robot_type_list[r[1]][2][j]

            # 所有机器人的总能力要大于每个类型的任务需要的能力
            if task_capacity > robot_capacity:
                task_type_list[i][2][j] = robot_capacity / (max_robot_num + random.uniform(0, 0.05))
                task_type_list[i][2][j] = round(task_type_list[i][2][j], 4)

    # 2 资源
    for i in range(len(task_type_list)):
        for j in range(len(task_type_list[i][3])):
            task_resource = task_type_list[i][3][j]
            robot_resource = 0
            for r in robot_list:
                robot_resource += robot_type_list[r[1]][3][j]

            # 所有机器人的总资源要大于每个类型的任务需要的资源
            if task_resource > robot_resource:
                task_type_list[i][3][j] = robot_resource / (max_robot_num + random.uniform(0, 0.05))
                task_type_list[i][3][j] = round(task_type_list[i][3][j], 4)

    return robot_type_list, task_type_list


def generate_all_data(num_capacity, num_resource, num_robot, num_robot_type, min_v, max_v,
                      num_task, num_task_type, min_t, max_t, x_range, y_range, num_constraint):
    """
    随机生成所有的实验数据

    Returns:
        (list) 机器人类型列表; (list) 任务类型列表; (list) 机器人列表; (list) 任务列表
    """
    # 生成机器人类型列表
    robot_type_list = generate_robot_type(num_robot_type, num_capacity, num_resource, min_v, max_v)

    # 生成任务类型列表
    task_type_list = generate_task_type(num_task_type, num_capacity, num_resource, min_t, max_t)

    # 生成机器人列表
    robot_list = generate_robot(num_robot, num_robot_type)

    # 生成任务列表
    task_list = generate_task(num_task, num_task_type, x_range, y_range, num_constraint)

    # 检查并修复
    while not check_data(robot_list, task_list, robot_type_list, task_type_list):
        robot_type_list, task_type_list = repair_data(robot_list, task_list, robot_type_list, task_type_list, num_capacity, num_resource)

    return robot_type_list, task_type_list, robot_list, task_list


def save_data_2_file(path, robot_type_list, task_type_list, robot_list, task_list, num_constraint, num_capacity,
                     num_resource, exp_num):
    """
    将生成的实验数据保存到json文件中

    Args:
        path:文件保存目录
        robot_type_list:机器人类型列表
        task_type_list:任务类型列表
        robot_list:机器人列表
        task_list:任务列表
        num_constraint:先序约束数量
        num_capacity:能力数量
        num_resource:资源数量
        exp_num:实验编号
    """
    if not os.path.exists(path):
        os.mkdir(path)
    file_path = path+'exp'+str(exp_num)+'_R'+str(len(robot_list))+'_T'\
                +str(len(task_list))+'_RT'+str(len(robot_type_list))+'_TT'\
                +str(len(task_type_list))+'_P'+str(num_constraint)+\
                '_CN'+str(num_capacity)+'_RN'+str(num_resource)+'.json'

    # 将所有的数据存入同一个字典, 解析成一个json对象
    data = {'robot': robot_list,
            'task': task_list,
            'robot type': robot_type_list,
            'task type': task_type_list,
            'capacity': num_capacity,
            'resource': num_resource}

    with open(file_path, 'w') as f:
        f.write(json.dumps(data))
    f.close()


def read_data_from_file(path) -> object:
    """
    从json文件中读取实验数据

    Args:
        path:json文件目录

    Returns:
        (list) 机器人类型列表; (list) 任务类型列表; (list) 机器人列表; (list) 任务列表; (int) 能力数量; (int) 资源数量
    """
    with open(path) as f:
        data = json.load(f)
    f.close()

    return data['robot type'], data['task type'], data['robot'], data['task'], data['capacity'], data['resource']


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    exp_num = 30

    task_nums =         [100, 300, 500]
    robot_nums =        [5, 5, 5, 5, 10, 10, 10, 10, 15, 20]
    robot_type_nums =   [2, 3, 4, 5, 2, 3, 4, 5, 5, 5]
    task_type_nums =    [5, 10, 15, 20, 5, 10, 15, 20, 20, 20]
    resource_nums =     [2, 3, 4, 5, 2, 3, 4, 5, 5, 5]
    capacity_nums =     [2, 3, 4, 5, 2, 3, 4, 5, 5, 5]
    constraint_rate =   [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8]

    id = 0
    for i in range(len(task_nums)):
        for j in range(len(robot_nums)):

            print("生成第"+str(id)+"组实验数据......")
            constraint_num = int(task_nums[i] * constraint_rate[j])

            robot_type_list, task_type_list, robot_list, task_list = generate_all_data(num_robot=robot_nums[j],
                                                                                       num_task=task_nums[i],
                                                                                       num_robot_type=robot_type_nums[j],
                                                                                       num_task_type=task_type_nums[j],

                                                                                       num_constraint=constraint_num,
                                                                                       num_capacity=capacity_nums[j],
                                                                                       num_resource=resource_nums[j],

                                                                                       min_v=5,
                                                                                       max_v=10,
                                                                                       min_t=200,
                                                                                       max_t=500,
                                                                                       x_range=(-1000, 1000),
                                                                                       y_range=(-1000, 1000))

            save_data_2_file('exp_0515/', robot_type_list, task_type_list, robot_list, task_list, constraint_num,
                             capacity_nums[j], resource_nums[j], id)
            id += 1


