import copy
import csv
import scipy.io as sio

from resource_allocation_optimization import *
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import time
from problems.noop.problem_noop import noop_users,val_noop_users
import torch
from my_utils import *

def show_order_influence():

    y_max,y_min,y_mean=[],[],[]
    user_number_list=[2,3,4,5]
    user_number_list=[3]
    d_min,d_max=1,50
    for user_number in user_number_list:
        for _ in range(1):
            users=generate_topology(user_number,d_min,d_max)
            throughput_his=baseline_exhaustive_search(users,0,1e-8,need_throughput_his=True)
            throughput_his=np.asarray(throughput_his)
            print(throughput_his)
            print(f'max={throughput_his.max()},min={throughput_his.min()},mean={throughput_his.mean()}')
            y_max.append(throughput_his.max())
            y_min.append(throughput_his.min())
            y_mean.append(throughput_his.mean())
            y_max[-1]=(y_max[-1]-y_mean[-1])/y_mean[-1]
            y_min[-1]=(y_min[-1]-y_mean[-1])/y_mean[-1]
    plt.plot(user_number_list,y_max,label='y_max/y_mean')
    plt.plot(user_number_list,y_min,label='y_min/y_mean')
    # plt.plot(user_number_list,y_mean,label='y_mean/y_min')
    plt.legend()
    plt.xlabel('user number')
    plt.ylabel('throughput')
    plt.title(f'd\in$[{d_min,d_max}]$')
    plt.show()



def show_speed_performance_dataset(dataset):
    dataset=copy.deepcopy(dataset)
    users=copy.deepcopy(val_noop_users)

    # users=noop_users
    user_num= len(users)
    # print('user_num:',user_num)
    frames_num=dataset.data_num
    print('dataset_datanum:',dataset.data_num)
    # frames_num=len(dataset)

    users_g = dataset.g.squeeze(-1).numpy()
    users_w = dataset.w.squeeze(-1).numpy()
    # users_g=dataset
    methods={
        'baseline_g_order_ascd':baseline_g_order_asc,
             'baseline_g_order_desc':baseline_g_order_desc,
             'baseline_w_order_aesc': baseline_w_order_aesc,
             'baseline_w_order_desc':baseline_w_order_desc,
             'baseline_heuristic_method_qian':baseline_heuristic_method_qian,
             # 'baseline_random':baseline_random,
             # 'baseline_exhaustive_search':baseline_exhaustive_search
             }
    performance_his=defaultdict(list)
    speed_his=defaultdict(list)
    # baseline top 15
    top15_his = {}
    for name in methods:
        top15_his[name] = {'top15':0,'top10':0,'top5':0,'top1':0,}


    try:
        top15_list = sio.loadmat("Top15/n%d_top15_1" % (user_num))['performance_list']
    except:
        print('Waiting for exhaustive search')

    # exhaustive search top15 and save
    t_ten_performance_list = []
    t_ten_order_list = []
    for frame_i in tqdm(range(frames_num)):
        # tg=users_g[frame_i,:]
        tg = users_g[frame_i]
        tw = users_w[frame_i]
        for i in range(user_num):
            users[i].g= tg[i]
            users[i].w = tw[i]
        no_error=True
        t_speed_list=[]
        t_performance_list=[]
        for method_name,method in methods.items():
            # random.shuffle(users)
            time_start=time.time()
            # 穷搜存一下top15
            if method_name == 'baseline_exhaustive_search':
                # try:
                #     top15_list = sio.loadmat("Top15/n%d_top15_1" % (user_num))['performance_list']
                # except:
                #     no_error = False
                # if no_error:
                #     continue
                try:
                    _, t_throuhput, t_ten_throughput, t_ten_decode_order = method(users=users, noise=args.noise, alpha=args.alpha)
                except BaseException as e:
                    no_error = False
                    print(f'e={e}')
                    break
                time_end = time.time()
                t_speed_list.append(time_end - time_start)
                t_performance_list.append(t_throuhput)
                t_ten_performance_list.append(t_ten_throughput)
                t_ten_order_list.append(t_ten_decode_order)
            else:
                try:
                    _,t_throuhput=method(users=users,noise=args.noise,alpha=args.alpha)
                except BaseException as e:
                    no_error=False
                    print(f'e={e}')
                    break
                time_end=time.time()
                t_speed_list.append(time_end-time_start)
                t_performance_list.append(t_throuhput)
                # speed_his[method_name].append(time_end-time_start)
                # performance_his[method_name].append(t_throuhput)
                # './Top15/n%d_top15_1.mat' % (user_num)
                # try:
                #     top15_i = copy.deepcopy(top15_list[frame_i])
                #     top15_i = top15_i.tolist()
                #     top15_i.sort(reverse=True)
                #     if t_throuhput >= top15_i[14]:
                #         top15_his[method_name]['top15'] += 1
                #     if t_throuhput >= top15_i[9]:
                #         top15_his[method_name]['top10'] += 1
                #     if t_throuhput >= top15_i[4]:
                #         top15_his[method_name]['top5'] += 1
                #     if t_throuhput >= top15_i[0]:
                #         top15_his[method_name]['top1'] += 1
                # except BaseException as e:
                #     no_error = False
                #     print(f'e={e}, waiting')

        # if no_error:
        for i,method_name in enumerate(methods.keys()):
            # if method_name == 'baseline_exhaustive_search':
            #     continue
            speed_his[method_name].append(t_speed_list[i])
            performance_his[method_name].append(t_performance_list[i])

    # for method_name in methods.keys():
    print(f'Speed：')
    for k, v in speed_his.items():
        print(f'{k}:{sum(v) / len(v)}')
    # if no_error:
    #     th_sum_exhaustive = 0
    #     for i in range(len(top15_list)):
    #         th_sum_exhaustive += max(top15_list[i])
    #     print(f'baseline_exhaustive_search:',th_sum_exhaustive/len(top15_list))

    print(f'Performance：')
    for k, v in performance_his.items():
        print(f'{k}:{sum(v) / len(v)}')

    # for k, v in top15_his.items():
    #     if k != 'baseline_exhaustive_search':
    #         for top_num in v:
    #             print(f'{k}-{top_num}:{v[top_num] / len(dataset)}')
    # return speed_his,performance_his

    # if 'baseline_exhaustive_search' in methods and not no_error:
    #     # 把本轮做好的top15样本存好，留到下一轮对比。因为固定了随机种子，所以没问题！
    #     # file_name = './Top15/18_n%d_top15_1.csv'%(user_num)
    #     # with open(file_name,'w',encoding='utf-8') as f:
    #     #     csv_writer = csv.writer(f)
    #     #     csv_writer.writerows(t_ten_performance_list)
    #     #     csv_writer.writerows(t_ten_order_list)
    #     # print(t_ten_performance_list,t_ten_order_list)
    #     # 不能跑4个以下节点，因为不满15  3！=6 4！=24
    #     sio.savemat('./Top15/n%d_top15_1.mat'%(user_num), {'performance_list': t_ten_performance_list, 'order_list': t_ten_order_list})



# def show_random_performance_dataset


if __name__=='__main__':
    print(f'args.noise={args.noise}')
    # compare performance
    user_num = args.user_num
    frames_num = 100
    # 生成拓扑
    # users = generate_topology(user_num,1,20,1,3)
    users = noop_users
    # 用户在各帧的信道质量可以实时生成，也可以一次性生成完毕，这里先采用后者
    users_g_hat = np.asarray([tuser.g_hat for tuser in users])
    users_g = np.random.rayleigh(scale=1, size=[frames_num, user_num]) * users_g_hat
    # show_order_influence()