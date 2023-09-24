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
users_g_hat = get_users_g_hat(noop_users)
user_w_hat = get_users_w_hat(noop_users)
np.random.seed(6741)
g = np.random.rayleigh(1,size=[100,len(noop_users)])*users_g_hat
w = np.asarray([random.choices([1,2,4,8,16,32], k=4) for i in range(len(noop_users))])

w_to_1 = 1e5 # 使得输入网络的w接近于1  原来为1e9

# NOMA order optimal problem
class NOOP:

    NAME='noop'

    @staticmethod
    def get_costs(dataset,pi):
        # dataset.w
        # print('get_cost dataset',dataset)
        users=noop_users
        # print(dataset)
        # dataset.cpu().numpy() 存的是 [P_max,w, r1_real, r1_complex, r2_real, r2_complex]  如果增加天线数，一定要记得更改
        g=dataset.cpu().numpy()[:,:,-4:]  # 此处已经改过了，原为g=dataset.cpu().numpy()[:,:,-1]  现为[r1_real, r1_complex, r2_real, r2_complex]
        # print('g',g)
        w= dataset.cpu().numpy()[:,:,1]     # 原为w= dataset.cpu().numpy()[:,:,-2]
        # print(dataset.cpu())
        # print('dataset.cpu().numpy()',dataset.cpu().numpy())
        # print('dataset.H',H )
        # print('get_cost中的权重',w[0],w[1])  # 把生成拓扑函数中的tw换成w了，随机了
        # print('当前权重与信道增益',g,w)
        # print("g",g)
        decode_order=pi.cpu().numpy()
        reward_list=[]
        for t_g,t_w,t_decode_order in zip(g,w,decode_order):
            set_users_g(users,t_g/w_to_1)       # [实部1,虚部1,实部2,虚部2]
            set_users_w(users,t_w)
            users_order=sort_by_decode_order(users,t_decode_order)
            reward=get_max_sum_weighted_alpha_throughput(users=users_order)
            reward_list.append(-reward)
        return torch.tensor(reward_list,device=dataset.device),None
    """
    dataset
    tensor([[[ 1.0000,  8.0000,  7.7838],
         [ 1.0000, 32.0000,  3.6171],
         [ 1.0000, 16.0000,  0.8716],
         [ 1.0000, 16.0000,  1.0126]],

        [[ 1.0000,  8.0000,  6.2264],
         [ 1.0000, 32.0000,  0.1806],
         [ 1.0000, 16.0000,  0.8035],
         [ 1.0000, 16.0000,  0.4620]],

        [[ 1.0000,  8.0000,  2.8423],
         [ 1.0000, 32.0000,  2.3162],
         [ 1.0000, 16.0000,  0.3755],
         [ 1.0000, 16.0000,  0.4184]],

        ...,

        [[ 1.0000,  8.0000,  6.4023],
         [ 1.0000, 32.0000,  1.9694],
         [ 1.0000, 16.0000,  0.4532],
         [ 1.0000, 16.0000,  0.5055]],

        [[ 1.0000,  8.0000,  8.6839],
         [ 1.0000, 32.0000,  5.6186],
         [ 1.0000, 16.0000,  0.2338],
         [ 1.0000, 16.0000,  0.1123]],

        [[ 1.0000,  8.0000,  6.0686],
         [ 1.0000, 32.0000,  2.1613],
         [ 1.0000, 16.0000,  0.2942],
         [ 1.0000, 16.0000,  0.5981]]])
    """



    @staticmethod
    def make_dataset(*args,**kwargs):
        return NOOPDataset(*args,**kwargs)

    @staticmethod
    def load_dataset(*args, **kwargs):
        return LoadNOOPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args,**kwargs):
        return StateNOOP.initialize(*args,**kwargs)

    @staticmethod
    def beam_search(*args,**kwargs):
        return

class NOOPDataset(Dataset):
    # 以上为单天线生成拓扑
    # def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
    #     users=noop_users
    #     self.data_num=num_samples
    #     self.users_g_hat=get_users_g_hat(users)
    #     if seed:
    #         np.random.seed(seed)
    #     g=np.random.rayleigh(1,size=[num_samples,len(users)])*self.users_g_hat
    #     # 把权重也随机喂入
    #     values_w = [1,2,4,8,16,32]
    #     w = np.asarray([random.choices(values_w, k=len(users)) for i in range(num_samples)])
    #     # print("datasize w",w)
    #     self.g=torch.FloatTensor(g).unsqueeze(-1)
    #     self.w = torch.FloatTensor(w).unsqueeze(-1)

    # 现在咱们用多天线生成拓扑
    def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
        users=noop_users
        self.data_num=num_samples
        self.users_g_hat=get_users_g_hat(users)
        print('g_hat',self.users_g_hat,self.users_g_hat.shape)
        if seed:
            np.random.seed(seed)

        user_num = len(users)
        rx_1 = np.random.randn(num_samples, 2, user_num)
        rx_2 = np.random.randn(num_samples, 2, user_num)
        H_matrix =[]
        for i in range(len(rx_1)):
            rx_complex = (1/np.sqrt(user_num) *(np.vstack((rx_1[i,0,:],rx_2[i,0,:])) + np.vstack((rx_1[i,1,:],rx_2[i,1,:]))*1j) /np.sqrt(2))
            H_matrix.append(rx_complex)

        H_matrix = np.array(H_matrix)
        rx = np.concatenate((rx_1,rx_2),axis=1)

        # g() 天线1实部、天线1虚部、天线2实部、天线2虚部
        g=rx*np.sqrt(self.users_g_hat)   # 乘上了g_hat部分,此处是否应该开根号？ 所有关于信道增益的部分，我都应该先开根号 于是全换了。
        # print('g',g,g.shape)
        # 把权重也随机喂入
        values_w = [1,2,4,8,16,32]
        w = np.asarray([random.choices(values_w, k=len(users)) for i in range(num_samples)])
        # print("datasize w",w)

        # 单天线需要扩维，多天线直接够了
        # self.g=torch.FloatTensor(g).unsqueeze(-1)
        self.g = torch.FloatTensor(g).transpose(1, 2)
        self.w = torch.FloatTensor(w).unsqueeze(-1)
        self.H = np.transpose(H_matrix,(0,2,1))

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

# from resource_allocation_optimization import generate_topology
# users=generate_topology(5)
# dataset=NOOPDataset(users,10)
class LoadNOOPDataset(Dataset):
    # 以上为单天线生成拓扑
    # def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
    #     users=noop_users
    #     self.data_num=num_samples
    #     self.users_g_hat=get_users_g_hat(users)
    #     if seed:
    #         np.random.seed(seed)
    #     g=np.random.rayleigh(1,size=[num_samples,len(users)])*self.users_g_hat
    #     # 把权重也随机喂入
    #     values_w = [1,2,4,8,16,32]
    #     w = np.asarray([random.choices(values_w, k=len(users)) for i in range(num_samples)])
    #     # print("datasize w",w)
    #     self.g=torch.FloatTensor(g).unsqueeze(-1)
    #     self.w = torch.FloatTensor(w).unsqueeze(-1)

    # 现在咱们用多天线生成拓扑
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