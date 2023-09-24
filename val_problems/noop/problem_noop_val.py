import numpy as np
from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.noop.state_noop import StateNOOP
from utils.beam_search import beam_search
from my_utils import *
from resource_allocation_optimization import *
from conf import args
from tqdm import tqdm
import scipy.io as sio

seed_everything(args.seed)
# 关键的输入 input noop_users
noop_users = generate_topology(args.user_num, args.d_min, args.d_max, args.w_min, args.w_max)


w_to_1 = 1e5 # 使得输入网络的w接近于1  原来为1e9

# NOMA order optimal problem
class NOOP:

    NAME='noop'

    @staticmethod
    def get_costs(dataset,pi):
        # dataset.w
        # print('get_cost dataset',dataset)
        users=noop_users

        g=dataset.cpu().numpy()[:,:,-4:]
        # print('g',g)
        w= dataset.cpu().numpy()[:,:,1]

        decode_order=pi.cpu().numpy()
        reward_list=[]
        for t_g,t_w,t_decode_order in zip(g,w,decode_order):
            set_users_g(users,t_g/w_to_1)       # [实部1,虚部1,实部2,虚部2]
            set_users_w(users,t_w)
            users_order=sort_by_decode_order(users,t_decode_order)
            reward=get_max_sum_weighted_alpha_throughput(users=users_order)
            reward_list.append(-reward)
        return torch.tensor(reward_list,device=dataset.device),None


    @staticmethod
    def load_dataset(*args, **kwargs):
        return LoadNOOPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args,**kwargs):
        return StateNOOP.initialize(*args,**kwargs)

    @staticmethod
    def beam_search(*args,**kwargs):
        return


class LoadNOOPDataset(Dataset):

    def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
        users=noop_users
        self.data_num=num_samples
        self.users_g_hat=get_users_g_hat(users)
        print('g_hat',self.users_g_hat,self.users_g_hat.shape)
        if seed:
            np.random.seed(seed)

        val_g = sio.loadmat("Validation/n%d_valdataset.mat" % (size))['val_g']
        # print('val_g',val_g,type(val_g))
        val_w = sio.loadmat("Validation/n%d_valdataset.mat" % (size))['val_w']
        val_H = sio.loadmat("Validation/n%d_valdataset.mat" % (size))['val_H']
        # print("datasize w",w)

        # 单天线需要扩维，多天线直接够了
        # self.g=torch.FloatTensor(g).unsqueeze(-1)
        self.g = torch.FloatTensor(val_g)
        self.w = torch.FloatTensor(val_w)
        self.H = np.transpose(val_H)

        # 复数无法存成torch 这样存将导致丢失虚部
        # self.H = torch.FloatTensor(H_matrix).transpose(1, 2)

    def __len__(self):
        return self.data_num

    # dataset[idx] 取索引得到的返回值
    def __getitem__(self, idx):
        t=[]
        # print('net input',self.g[idx])
        ### 这里t.append([tuser.p_max,w,tg*w_to_1]) 换成了t.append([tuser.p_max,tw,tg*w_to_1]) 于是随机了
        for tuser,tg,tw in zip(noop_users,self.g[idx],self.w[idx]):
            t.append(np.hstack(([tuser.p_max,tw],tg*w_to_1)))
            # print('net input',np.hstack(([tuser.p_max,tw],tg*w_to_1)))
            # t.append([tuser.p_max,tw,tg*w_to_1])
        t=torch.FloatTensor(t)
        return t