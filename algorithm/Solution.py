# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/3/27 20:01
# File : Solution.py
import math
import numpy as np

from algorithm.General_Utils import get_alliance_by_id


class Solution:
    def __init__(self, robot_n, task_n, weight_vector):
        self.weight_vector = weight_vector
        self.robot_n = robot_n
        self.task_n = task_n
        # r_t_s: [[[task_id, x, y, alliance_id],...],[...],...]
        self.r_t_s = [[] for _ in range(robot_n)]
        # t_a_s: [[task_id, alliance_id],...]
        self.t_a_s = []
        # cost_finish_time: [[task_start_time, task_finish_time],...]
        self.start_finish_time = np.zeros(shape=(task_n, 2))
        # objs: [max_task_finish_time, average_robot_travel_time, average_task_start_time]
        self.objs = np.zeros(3)
        self.F = 0

        self.type = 0

    def add_one_task_alliance(self, task: list, alliance: list, robot_type_list: list):
        """
        添加一个(任务, 联盟)到解中

        Args:
            task: 当前要添加的任务
            alliance: 任务对应的联盟, [alliance_id, cost, alliance_robot_list]
            robot_type_list: 机器人类型列表, 提供机器人速度

        """
        # 1) 计算添加当前 任务-联盟 对应的 任务结束时间\机器人移动时间\任务开始时间
        task_finish_time, all_travel_time, task_start_time =\
            self.cpt_one_task_alliance_objs(task, alliance, robot_type_list)

        # 2) 更新各属性
        alliance_id = alliance[0]
        alliance_robot_list = alliance[2]

        self.t_a_s.append([task[0], alliance_id])
        for i in range(len(alliance_robot_list)):
            r = alliance_robot_list[i]
            self.r_t_s[r[0]].append([task[0], task[2], task[3], alliance_id])
        self.start_finish_time[task[0]] = [task_start_time, task_finish_time]
        objs_increment = self.cpt_objs_increment(task_finish_time, all_travel_time, task_start_time)    # 计算三个目标的增量
        self.objs += objs_increment
        self.F = np.dot(self.weight_vector, self.objs)

    def cpt_one_task_alliance_objs(self, task: list, alliance: list, robot_type_list: list):
        """
        计算添加当前 任务-联盟 后对应的 任务结束时间/机器人移动时间/任务开始时间

        Args:
            task:当前任务
            alliance:当前联盟
            robot_type_list:机器人类型列表

        Returns:
            (float) 任务结束时间, (np.array) 联盟各机器人的移动时间, (float) 任务开始时间

        """
        # 1) 判断任务的先序约束是否满足
        precedence_constraint = task[4]
        if len(precedence_constraint) == 0:
            max_pre_finish_time = 0
        else:
            if (self.start_finish_time[precedence_constraint, 1] == 0).any():
                assert False, "任务%d的先序任务未执行!" % task[0]
            max_pre_finish_time = np.max(self.start_finish_time[precedence_constraint, 1])

        # 2) 计算任务的三个时间
        alliance_cost = alliance[1]
        alliance_robot_list = alliance[2]
        alliance_robot_num = len(alliance_robot_list)

        last_task_finish_time = np.zeros(alliance_robot_num)  # 存储联盟中所有机器人上一个任务的结束时间
        all_travel_time = np.zeros(alliance_robot_num)        # 存储联盟中所有机器人的移动时间
        for i in range(alliance_robot_num):
            r = alliance_robot_list[i]
            r_v = robot_type_list[int(r[1])][1]

            if len(self.r_t_s[r[0]]) == 0:  # 起点
                last_task_x = 0
                last_task_y = 0
                last_task_finish_time[i] = 0
            else:
                last_task = self.r_t_s[r[0]][-1]
                last_task_x = last_task[1]                      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                last_task_y = last_task[2]
                last_task_finish_time[i] = self.start_finish_time[last_task[0]][1]

            travel_time = math.sqrt((last_task_x - task[2]) ** 2 + (last_task_y - task[3]) ** 2) / r_v  # 机器人移动时间
            all_travel_time[i] = travel_time
        task_start_time = max(np.max(last_task_finish_time + all_travel_time), max_pre_finish_time)     # 任务开始时间
        task_finish_time = task_start_time + alliance_cost                                              # 任务结束时间

        return task_finish_time, all_travel_time, task_start_time

    def cpt_objs_increment(self, task_finish_time: float, all_travel_time: np.ndarray, task_start_time: float):
        """
        计算三个目标值的增量

        Args:
            task_finish_time:新任务的结束时间
            all_travel_time:新 任务-联盟 的机器人移动时间
            task_start_time:新任务的开始时间

        Returns:
            (np.array) 三个目标的增量

        """
        result = np.zeros(len(self.objs))
        result[0] = max(self.objs[0], task_finish_time) - self.objs[0]
        result[1] = np.sum(all_travel_time) / self.robot_n
        result[2] = task_start_time / self.task_n
        return result

    def set_new_tas(self, new_t_a_s, task_list, task_type_list, robot_list, robot_type_list):
        """
        根据新的tas构建解

        Args:
            new_t_a_s:新的任务-联盟执行序列
            task_list:任务列表
            t_a_cost:任务-联盟成本(dict)
            robot_type_list:机器人类型列表

        """
        for ta in new_t_a_s:
            task = task_list[ta[0]]
            task_type = task_type_list[task[1]]
            alliance_id = ta[1]
            alliance = get_alliance_by_id(alliance_id, task_type, robot_list, robot_type_list)      # ke yi you hua
            self.add_one_task_alliance(task, alliance, robot_type_list)

    def is_dominate(self, s):
        """
        判断当前解是否支配s

        Returns:
            (bool) 是否支配解s

        """
        count = 0
        for i in range(len(self.objs)):
            if self.objs[i] > s.objs[i]:
                return False
            elif self.objs[i] < s.objs[i]:
                count += 1
        if count > 0:
            return True
        else:
            return False


