# -*- coding: utf-8 -*-
# Author : Qian Tong
# Date : 2023/5/31 0:21
# File : Multiprocess_Run.py
import multiprocessing
import os
import random
import time
from datetime import datetime

import numpy as np

from algorithm.CACS.CACS import CACS
from data.Data_generator import read_data_from_file
from Utils import save_data

Params = {
    "CACS": {
        "pop_num": 3,
        "single_pop_size": 20,
        "alpha": 1,
        "beta": 2,
        "select_r": 0.9,
        "local_rho": 0.1,
        "global_rho": 0.1,
        "crossover_rate": 1.0,
        "mutation_rate": 0.1,
    }
}

Data_files = ['I_{1}_R5_T100_RT2_TT5_P20_CN2_RN2.json',
              'I_{2}_R5_T100_RT3_TT10_P40_CN3_RN3.json',
              'I_{3}_R5_T100_RT4_TT15_P60_CN4_RN4.json',
              'I_{4}_R5_T100_RT5_TT20_P80_CN5_RN5.json',
              'I_{5}_R10_T100_RT2_TT5_P20_CN2_RN2.json',
              'I_{6}_R10_T100_RT3_TT10_P40_CN3_RN3.json',
              'I_{7}_R10_T100_RT4_TT15_P60_CN4_RN4.json',
              'I_{8}_R10_T100_RT5_TT20_P80_CN5_RN5.json',
              'I_{9}_R15_T100_RT5_TT20_P80_CN5_RN5.json',
              'I_{10}_R20_T100_RT5_TT20_P80_CN5_RN5.json',
              'I_{22}_R5_T500_RT3_TT10_P200_CN3_RN3.json',
              'I_{11}_R5_T300_RT2_TT5_P60_CN2_RN2.json',
              'I_{12}_R5_T300_RT3_TT10_P120_CN3_RN3.json',
              'I_{13}_R5_T300_RT4_TT15_P180_CN4_RN4.json',
              'I_{14}_R5_T300_RT5_TT20_P240_CN5_RN5.json',
              'I_{15}_R10_T300_RT2_TT5_P60_CN2_RN2.json',
              'I_{16}_R10_T300_RT3_TT10_P120_CN3_RN3.json',
              'I_{17}_R10_T300_RT4_TT15_P180_CN4_RN4.json',
              'I_{18}_R10_T300_RT5_TT20_P240_CN5_RN5.json',
              'I_{19}_R15_T300_RT5_TT20_P240_CN5_RN5.json',
              'I_{20}_R20_T300_RT5_TT20_P240_CN5_RN5.json',
              'I_{21}_R5_T500_RT2_TT5_P100_CN2_RN2.json',
              'I_{23}_R5_T500_RT4_TT15_P300_CN4_RN4.json',
              'I_{24}_R5_T500_RT5_TT20_P400_CN5_RN5.json',
              'I_{25}_R10_T500_RT2_TT5_P100_CN2_RN2.json',
              'I_{26}_R10_T500_RT3_TT10_P200_CN3_RN3.json',
              'I_{27}_R10_T500_RT4_TT15_P300_CN4_RN4.json',
              'I_{28}_R10_T500_RT5_TT20_P400_CN5_RN5.json',
              'I_{29}_R15_T500_RT5_TT20_P400_CN5_RN5.json',
              'I_{30}_R20_T500_RT5_TT20_P400_CN5_RN5.json', ]

# 解评估次数实验
def eval_exp_worker(alg, max_eval_num, data_files, seed):
    for data_file in data_files:
        name = "%s_Eval%d" % (alg, max_eval_num)
        instance_name = data_file.split(".")[0]

        data_file_settings = data_file.split('_')
        rob_num = int(data_file_settings[2][1:])
        task_num = int(data_file_settings[3][1:])
        max_run_time = rob_num * task_num
        data_file_path = "./data/exp_30/%s" % data_file
        save_path = "./result/%s_%s/%s/" % (data_file_settings[0], data_file_settings[1], alg)
        rt_list, tt_list, r_list, t_list, n_c, n_r = read_data_from_file(path=data_file_path)

        random.seed(seed)
        np.random.seed(seed)
        start_timestamp = time.time()
        start_local_time = time.localtime(start_timestamp)
        start_actual_time = time.strftime('%Y-%m-%d %H:%M:%S', start_local_time)
        print("====================== %s Alg: %s Exp: %s Seed: %d ======================"
              % (start_actual_time, alg, data_file, seed))

        algorithm = CACS(robot_list=r_list,
                         task_list=t_list,
                         robot_type_list=rt_list,
                         task_type_list=tt_list,
                         num_capacity=n_c,
                         num_resource=n_r,
                         pop_num=Params['CACS']['pop_num'],
                         single_pop_size=Params['CACS']['single_pop_size'],
                         alpha=Params['CACS']['alpha'],
                         beta=Params['CACS']['beta'],
                         select_r=Params['CACS']['select_r'],
                         local_rho=Params['CACS']['local_rho'],
                         global_rho=Params['CACS']['global_rho'],
                         crossover_rate=Params['CACS']['crossover_rate'],
                         mutation_rate=Params['CACS']['mutation_rate'],
                         max_run_time=max_run_time,
                         max_eval_num=max_eval_num)

        try:
            start_time = time.process_time()
            _ = algorithm.init()
            archive, _ = algorithm.run_eval(name, instance_name, seed)
            end_time = time.process_time()
            # 保存实验结果
            # 1) 算法运行的CPU时间
            run_CPU_time = end_time - start_time
            # 2) 算法得到的 archive 中每个个体的目标函数值
            archive_objs = [s.objs.tolist() for s in archive]
            data = {'run_CPU_time': run_CPU_time, 'all_archive_objs': archive_objs}
            # 3) 算法在每个目标上的最优解
            np_archive_objs = np.array(archive_objs)
            single_obj_min_index = np.argmin(np_archive_objs, axis=0)
            for obj in range(len(single_obj_min_index)):
                data['solution_min_obj' + str(obj) + '_rts'] = archive[single_obj_min_index[obj]].r_t_s
                data['solution_min_obj' + str(obj) + '_tas'] = archive[single_obj_min_index[obj]].t_a_s
                data['solution_min_obj' + str(obj) + '_objs'] = archive[single_obj_min_index[obj]].objs.tolist()
            result_path = save_path + datetime.now().strftime('%m%d-%H%M%S') + "_" + str(seed)
            os.mkdir(result_path)
            save_data(result_path + './result.pickle', data)
        except Exception as e:
            print("Exception Info: ", repr(e))


if __name__ == '__main__':
    Seeds = np.random.randint(0, 9999, 30)
    args = []
    max_eval_num = 10000
    for seed in Seeds:
        args.append(("CACS", max_eval_num, Data_files, seed))
    pool = multiprocessing.Pool(30)
    pool.starmap_async(eval_exp_worker, args)
    # 关闭进程池
    pool.close()
    # 等待进程池中的进程结束
    pool.join()
