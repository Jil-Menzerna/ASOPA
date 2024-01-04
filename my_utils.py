import random
import os
import numpy as np
import torch


def seed_everything(seed=3258):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_users_g(users, tg_list):
    assert len(users) == len(tg_list), '信道信息数量与用户数不匹配'
    for user, tg in zip(users, tg_list):
        user.g = tg

def set_users_w(users, tw_list):
    assert len(users) == len(tw_list), '权重信息数量与用户数不匹配'
    for user, tw in zip(users, tw_list):
        user.w = tw


def get_usrs_g(users):
    users_g=np.asarray([tuser.g for tuser in users])
    return users_g


def get_users_g_hat(users):
    users_g_hat = np.asarray([tuser.g_hat for tuser in users])
    return users_g_hat

def get_users_w_hat(users):
    users_w_hat = np.asarray([tuser.w_hat for tuser in users])
    return users_w_hat


def random_set_users_g(users):
    """
    按照瑞利分布来随机设置用户的信道质量
    """
    users_g_hat = get_users_g_hat(users)
    users_g = np.random.rayleigh(1, size=[len(users)]) * users_g_hat
    set_users_g(users, users_g)
