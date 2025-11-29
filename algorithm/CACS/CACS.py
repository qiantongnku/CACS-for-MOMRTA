# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/4/24 20:54
# File : CACS.py
import random
import time

import numpy as np

from data.Data_generator import read_data_from_file
from algorithm.General_Utils import update_archive, log_eval
from algorithm.CACS.CACS_Local_Search import local_search_eval
from algorithm.CACS.CACS_Sol_Const import generate_new_solution
from algorithm.CACS.utils.Pheromone_Utils import init_pheromone, local_update_pheromone, global_update_pheromone
from algorithm.CACS.utils.Heuristic_Utils import cal_heuristic_1


class CACS:
    def __init__(self,
                 robot_list,
                 task_list,
                 robot_type_list,
                 task_type_list,
                 num_capacity,
                 num_resource,
                 pop_num,
                 single_pop_size,
                 alpha,
                 beta,
                 select_r,
                 local_rho,
                 global_rho,
                 crossover_rate,
                 mutation_rate,
                 max_run_time,
                 max_eval_num):
        self.r_list = robot_list  # 机器人列表
        self.t_list = task_list  # 任务列表
        self.robot_num = len(self.r_list)  # 机器人数量
        self.task_num = len(self.t_list)  # 任务数量
        self.rt_list = robot_type_list  # 机器人类型列表
        self.tt_list = task_type_list  # 任务类型列表
        self.c_num = num_capacity  # 能力数量
        self.r_num = num_resource  # 资源数量

        self.pop_num = pop_num  # 种群数量
        self.single_pop_size = single_pop_size  # 单个种群大小
        self.alpha = alpha  # 信息素指数
        self.beta = beta  # 启发式信息指数
        self.select_r = select_r  # 选择控制参数
        self.local_rho = local_rho  # 局部更新信息素蒸发率
        self.global_rho = global_rho  # 全局更新信息素蒸发率
        self.crossover_rate = crossover_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率

        self.max_run_time = max_run_time  # 最大运行时间
        self.max_eval_num = max_eval_num  # 最大解评估次数

        self.archive = []  # 非支配解集
        self.weight_vectors = np.zeros(shape=(self.pop_num, self.pop_num))  # 单目标种群的权重向量
        self.pheromone_0 = np.zeros(self.pop_num)  # 各种群信息素初始值
        self.pheromone_1 = np.zeros(shape=(self.pop_num, self.task_num + 1, self.task_num))  # 任务-任务阶段 信息素
        self.pheromone_2 = np.zeros(shape=(self.pop_num, self.task_num, self.robot_num))  # 任务-机器人阶段 信息素
        self.heuristic_1 = np.zeros(shape=(self.task_num + 1, self.task_num))  # 任务-任务阶段 启发式信息

    # ########################### #
    #            初始化            #
    # ########################### #
    def init(self):
        start_time = time.process_time()
        # (1) 初始化信息素
        self.weight_vectors, self.pheromone_0, self.pheromone_1, self.pheromone_2 = \
            init_pheromone(self.pop_num, self.t_list, self.tt_list, self.r_list, self.rt_list, self.robot_num,
                           self.task_num)
        # (2) 初始化 任务-任务阶段 启发式信息
        self.heuristic_1 = cal_heuristic_1(self.t_list, self.task_num)
        end_time = time.process_time()
        return end_time - start_time

    # ########################### #
    #           种群进化           #
    # ########################### #
    # 以最大解评估次数做为终止条件
    def run_eval(self, name, instance_name, seed_num):
        # 记录实验开始的实际时间点 和 CPU时间点
        start_timestamp = time.time()
        start_local_time = time.localtime(start_timestamp)
        start_actual_time = time.strftime('%Y-%m-%d %H:%M:%S', start_local_time)
        start_time = time.process_time()

        total_eval_num = 0
        count = 0
        while True:
            count += 1

            # 1) 串行进化每个蚁群
            all_pop = []
            for p in range(self.pop_num):
                for i in range(self.single_pop_size):
                    if total_eval_num >= self.max_eval_num:
                        break
                    # 生成新解
                    new_s = generate_new_solution(self.t_list, self.tt_list, self.r_list, self.rt_list,
                                                  self.weight_vectors, self.pheromone_1, self.pheromone_2,
                                                  self.heuristic_1, self.alpha, self.beta, self.select_r, p)
                    total_eval_num += 1  # 解评估一次
                    all_pop.append(new_s)
                    # 信息素局部更新
                    self.pheromone_1[p], self.pheromone_2[p] = local_update_pheromone(new_s,
                                                                                      self.local_rho,
                                                                                      self.pheromone_0[p],
                                                                                      self.pheromone_1[p],
                                                                                      self.pheromone_2[p],
                                                                                      self.r_list)
            # 2) 使用蚁群更新Archive集
            _ = update_archive(all_pop, self.archive)
            # 3) 交叉变异得到新的种群
            if total_eval_num < self.max_eval_num:
                new_pop, total_eval_num = local_search_eval(total_eval_num, self.max_eval_num,
                                                            self.single_pop_size, self.archive, self.crossover_rate,
                                                            self.mutation_rate,
                                                            self.t_list, self.tt_list, self.r_list, self.rt_list,
                                                            self.robot_num, self.task_num, self.pop_num)
                # 4) 使用交叉变异种群更新Archive集
                if len(new_pop) > 0:
                    _ = update_archive(new_pop, self.archive)

            # 5) 判断是否终止
            if total_eval_num >= self.max_eval_num:
                break
            # 6) 信息素全局更新
            self.pheromone_1, self.pheromone_2 = global_update_pheromone(self.pop_num, self.archive, self.global_rho,
                                                                         self.pheromone_1, self.pheromone_2,
                                                                         self.r_list)

        end_time = time.process_time()
        end_timestamp = time.time()
        end_local_time = time.localtime(end_timestamp)
        end_actual_time = time.strftime('%Y-%m-%d %H:%M:%S', end_local_time)

        # 写入日志
        log_eval(name, self.max_eval_num, instance_name, seed_num, start_time, end_time, start_timestamp, end_timestamp,
                  start_actual_time, end_actual_time, count, self.archive)

        # 输出
        return self.archive, end_time - start_time


if __name__ == '__main__':
    random.seed(44)
    np.random.seed(44)

    data_path = "../../data/test_R3_T10_RT2_TT4_P9_CN2_RN2.json"

    rt_list, tt_list, r_list, t_list, n_c, n_r = read_data_from_file(path=data_path)
    algorithm = CACS(robot_list=r_list,
                     task_list=t_list,
                     robot_type_list=rt_list,
                     task_type_list=tt_list,
                     num_capacity=n_c,
                     num_resource=n_r,

                     pop_num=3,
                     single_pop_size=20,
                     alpha=1,
                     beta=2,
                     select_r=0.9,
                     local_rho=0.1,
                     global_rho=0.1,
                     crossover_rate=1.0,
                     mutation_rate=0.1,

                     max_run_time=10,
                     max_eval_num=10000)
    init_time = algorithm.init()
    result, run_time = algorithm.run_eval('test', 'test', 44)
