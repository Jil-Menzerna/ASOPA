import random

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
noop_users = generate_topology(args.user_num, args.d_min, args.d_max, args.w_min, args.w_max)
if args.user_num != args.val_user_num:          # if that, valdataset no padding 0
    val_noop_users = generate_val_topology(args.val_user_num, args.d_min, args.d_max, args.w_min, args.w_max)
else:
    val_noop_users = noop_users

users_g_hat = get_users_g_hat(noop_users)
user_w_hat = get_users_w_hat(noop_users)
np.random.seed(6741)
g = np.random.rayleigh(1,size=[100,len(noop_users)])*users_g_hat
w = np.asarray([random.choices([1,2,4,8,16,32], k=4) for i in range(len(noop_users))])

w_to_1 = 1e9 # let input close to 1

# NOMA order optimal problem
class NOOP:

    NAME='noop'

    @staticmethod
    def get_costs(dataset,pi):
        # print('dataset',dataset)
        # dataset.w
        # print('get_cost dataset',dataset)
        g = dataset.cpu().numpy()[:, :, -1]
        w = dataset.cpu().numpy()[:, :, -2]
        # print('gw',g[0],w[0],g[-5],w[-5])
        # print('len(dataset.cpu().numpy()[0])',len(dataset.cpu().numpy()[0]),len(noop_users))
        if len(dataset.cpu().numpy()[0]) == len(noop_users):
            users=noop_users
        else:
            users= val_noop_users

        decode_order=pi.cpu().numpy()
        reward_list=[]
        for t_g,t_w,t_decode_order in zip(g,w,decode_order):
            set_users_g(users,t_g/w_to_1)
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
        return NOOP_allnum_Dataset(*args,**kwargs)

    @staticmethod
    def load_val_dataset(*args, **kwargs):
        return NOOPValDataset(*args, **kwargs)

    @staticmethod
    def make_allnum_dataset(*args,**kwargs):
        return NOOP_allnum_Dataset(*args,**kwargs)

    @staticmethod
    def make_state(*args,**kwargs):
        return StateNOOP.initialize(*args,**kwargs)

    @staticmethod
    def beam_search(*args,**kwargs):
        return

class NOOPDataset(Dataset):

    def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
        print('size',size)
        if size != args.user_num:
            users= val_noop_users
        else:
            users=noop_users
        self.data_num=num_samples
        self.users_g_hat=get_users_g_hat(users)
        if seed:
            np.random.seed(seed)
        g=np.random.rayleigh(1,size=[num_samples,len(users)])*self.users_g_hat
        ng = g + np.random.normal(loc=0,scale=1,size=[num_samples,len(users)])*args.noise
        # shuffle weight
        values_w = [1,2,4,8,16,32]
        w = np.asarray([random.choices(values_w, k=len(users)) for i in range(num_samples)])
        # print("datasize w",w)
        self.g=torch.FloatTensor(g).unsqueeze(-1)
        # print('training_g',self.g)
        self.w = torch.FloatTensor(w).unsqueeze(-1)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        t=[]
        for tuser,tg,tw in zip(noop_users,self.g[idx],self.w[idx]):
            t.append([tuser.p_max,tw,tg*w_to_1])
        t=torch.FloatTensor(t)
        return t


class NOOPValDataset(Dataset):

    def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
        # print('size', size)
        if size != args.user_num:
            users = val_noop_users
        else:
            users = noop_users
        self.data_num = num_samples
        self.users_g_hat = get_users_g_hat(users)
        if seed:
            np.random.seed(seed)

        val_g = sio.loadmat("Val/n%d_valdataset.mat" % (size))['val_g']
        # print('val_g',val_g,type(val_g))
        val_w = sio.loadmat("Val/n%d_valdataset.mat" % (size))['val_w']
        # print('val_g size',val_g.size())


        # print("datasize w",w)
        self.g = torch.FloatTensor(val_g)
        self.w = torch.FloatTensor(val_w)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        t=[]
        for tuser,tg,tw in zip(noop_users,self.g[idx],self.w[idx]):
            t.append([tuser.p_max,tw,tg*w_to_1])
        t=torch.FloatTensor(t)
        return t



# from resource_allocation_optimization import generate_topology
# users=generate_topology(5)
# dataset=NOOPDataset(users,10)
class NOOP_allnum_Dataset(Dataset):
    def __init__(self,num_samples=1000,seed=1234,size=len(noop_users),filename=None,distribution=None):
        users_list = []
        g_list = []
        w_list =[]
        self.data_num = num_samples

        # num_list = [i for i in range(args.num_min,args.num_max+1)]  # 5-10
        num_list = [i for i in range(args.num_min,args.num_max+2,2)]  # 10-20

        values_w = [1, 2, 4, 8, 16, 32]
        list_num_random = [213,213,213,213,214,214]
        random.shuffle(list_num_random)
        for user_num in num_list:
            print('users_num:',user_num)
            tusers_list=generate_topology(user_num, args.d_min, args.d_max, args.w_min, args.w_max,args.num_max) # 5 6 7 8 9 10的 [Usr1,2,3,4,5]
            users_list.append(tusers_list)
            print('users_list',len(users_list))
        for i in range(len(users_list)):
            print(i)
            tusers_list = users_list[i]
            self.users_g_hat= get_users_g_hat(tusers_list)
            self.users_g_hat = self.users_g_hat[self.users_g_hat != 0]
            if seed:
                np.random.seed(seed)
            user_num = 0
            # print('tusers_list.g',[tusers.g for tusers in tusers_list])
            for g_ in [tusers.g for tusers in tusers_list]:
                if g_ > 0:
                    user_num += 1
            list_num = list_num_random[i]
            g= np.hstack((np.random.rayleigh(1,size=[list_num,user_num])*self.users_g_hat, np.zeros([list_num,len(tusers_list)-user_num])))
            w = np.hstack((np.array([random.choices(values_w, k=user_num) for i in range(list_num)]), np.zeros([list_num,len(tusers_list)-user_num])))
            tg_list = list(g)
            tw_list = list(w)
            g_list.extend(tg_list)
            w_list.extend(tw_list)

        state_list = [[g_list[i],w_list[i]] for i in range(len(g_list))]
        random.shuffle(state_list)
        g_random = []
        w_random = []
        for i in range(len(state_list)):
            g_r,w_r = state_list[i]
            # print('g-w',g_r,w_r)
            g_random.append(g_r)
            w_random.append(w_r)

        # print('g_list',g_list)
        # print('g_list',g_random[0])
        # print('w_list',w_random[0])

        # 把权重也随机喂入
        g_random=np.array(g_random)
        w_random=np.array(w_random)
        # print("datasize w",w)
        self.g=torch.FloatTensor(g_random).unsqueeze(-1)
        self.w = torch.FloatTensor(w_random).unsqueeze(-1)
        # self.g=  g_random
        # self.w = w_random

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        t=[]
        for tuser,tg,tw in zip(noop_users,self.g[idx],self.w[idx]):
            if tw and tg:
                t.append([tuser.p_max,tw,tg*w_to_1])
            else:                                       # else: padding 0
                t.append([0, tw, tg * w_to_1])
        t=torch.FloatTensor(t)
        return t