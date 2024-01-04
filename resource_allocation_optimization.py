# 构建场景；求解在给定解码顺序下的资源分配问题
import numpy as np
# import cvxpy as cp
from cvxopt import matrix, solvers
import random
import math
import copy
from itertools import permutations
# import nlopt
# import tianshou
import keyword
from conf import args

# __all__=()
solvers.options['show_progress'] = False


class User:
    def __init__(self, tid, tdecode_order=-1, tp_max=1., tg=0.5, tw=1, td=1):
        self.id = tid  # 编号
        self.decode_order = tdecode_order  # 解码顺序
        self.p_max = tp_max  # 最大的发射功率
        self.g_hat = tg  # 上行平均信道增益，在每个帧都要乘以一个服从瑞利分布的随机数
        self.w_hat = tw
        self.g = tg
        self.w = tw  # 吞吐量权重
        self.d = td  # 到基站的距离


# generate_topology return user-list
def generate_topology(user_number=args.user_num, d_min=args.d_min, d_max=args.d_max, w_min=args.w_min,
                      w_max=args.w_max,max_num=10 ):

    d = [(td+1) * (d_max - d_min) / (user_number) + d_min for td in range(user_number)]

    # d = np.random.random([user_number]) * (d_max - d_min) + d_min
    g = [4.11 * (3e8 / (4 * math.pi * 915e6 * td)) ** 2.8 for td in d]
    print(f'\ng={g}')

    w = np.ones(user_number)

    w_possible = [1, 2, 4, 8, 16, 32] # 现在的基准是这个

    w_index = np.random.randint(0, len(w_possible), size=user_number)
    w = [w_possible[index] for index in w_index]

    p_max_possible = [1]
    p_max_index = np.random.randint(0, len(p_max_possible), size=user_number)
    p_max = [p_max_possible[index] for index in p_max_index]

    print(f'd={d}')
    # random.shuffle(w)
    print(f'w={w}')
    # random.shuffle(p_max)
    print(f'p_max={p_max}')

    users = []
    for i in range(user_number):
        users.append(User(tid=i, tp_max=p_max[i], tg=g[i], tw=w[i], td=d[i]))
    for i in range(user_number,max_num):
        users.append(User(tid=i, tp_max=0, tg=0, tw=0, td=0))
    # random.shuffle(users)
    # users=users[::-1]
    return users

# generate_topology for validating, return user-list
def generate_val_topology(user_number=args.user_num, d_min=args.d_min, d_max=args.d_max, w_min=args.w_min,
                      w_max=args.w_max,max_num=10 ):

    d = [(td+1) * (d_max - d_min) / (user_number) + d_min for td in range(user_number)]

    g = [4.11 * (3e8 / (4 * math.pi * 915e6 * td)) ** 2.8 for td in d]
    print(f'\ng={g}')

    w = np.ones(user_number)

    # w_possible = [1, 4, 16]
    w_possible = [1, 2, 4, 8, 16, 32] # 现在的基准是这个


    w_index = np.random.randint(0, len(w_possible), size=user_number)
    w = [w_possible[index] for index in w_index]

    p_max_possible = [1]
    p_max_index = np.random.randint(0, len(p_max_possible), size=user_number)
    p_max = [p_max_possible[index] for index in p_max_index]

    print(f'd={d}')
    # random.shuffle(w)
    print(f'w={w}')
    # random.shuffle(p_max)
    print(f'p_max={p_max}')

    users = []
    for i in range(user_number):
        users.append(User(tid=i, tp_max=p_max[i], tg=g[i], tw=w[i], td=d[i]))
    # random.shuffle(users)
    # users=users[::-1]
    return users


# 根据解码顺序对用户进行排序,如果不给定解码顺序就随机排序
def sort_by_decode_order(users=[], decode_order=None, need_order=False):
    if decode_order is None:
        decode_order = list(range(len(users)))
        random.shuffle(decode_order)
    users_order = [users[i] for i in decode_order]
    if need_order:
        return users_order, decode_order
    return users_order



# baseline channel gain ascend
def baseline_g_order_asc(users=[], alpha=None, noise=args.noise):
    a = [(i, tuser.g) for i, tuser in enumerate(users)]
    a = sorted(a, key=lambda ta: ta[1])
    decode_order = [i for i, _ in a]
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# baseline channel gain descend
def baseline_g_order_desc(users=[], alpha=None, noise=args.noise):
    a = [(i, tuser.g) for i, tuser in enumerate(users)]
    a = sorted(a, key=lambda ta: -ta[1])
    decode_order = [i for i, _ in a]
    # print(f'{decode_order}')
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# baseline weight descend
def baseline_w_order_desc(users=[], alpha=None, noise=args.noise):
    a = [(i, tuser.w) for i, tuser in enumerate(users)]
    a = sorted(a, key=lambda ta: -ta[1])
    decode_order = [i for i, _ in a]
    # print(f'{decode_order}')
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# baseline weight ascend
def baseline_w_order_aesc(users=[], alpha=None, noise=args.noise):
    a = [(i, tuser.w) for i, tuser in enumerate(users)]
    a = sorted(a, key=lambda ta: ta[1])
    decode_order = [i for i, _ in a]
    # print(f'{decode_order}')
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# baseline Qian's heuristic method
def baseline_heuristic_method_qian(users=[], alpha=None, noise=args.noise, need_random=False):

    if need_random:
        users_cun = copy.deepcopy(users)
        random.shuffle(users)
    tusers = copy.deepcopy(users)
    users_order = []
    user_number = len(users)
    for i in range(user_number):
        t = -1
        temp = -1
        for j in range(len(users_order) + 1):
            users_order.insert(j, tusers[i])
            tvalue = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha, noise=noise)
            if tvalue > temp:
                t = j
                temp = tvalue
            users_order.pop(j)
        users_order.insert(t, tusers[i])
    decode_order = [tuser.id for tuser in users_order]
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    if need_random:
        users = users_cun
    return decode_order, max_sum_weighted_alpha_throughput



def baseline_heuristic_method_qian_random(users=[], alpha=None, noise=args.noise):
    return baseline_heuristic_method_qian(users,alpha,noise,need_random=True)

# baseline exhaustive search
def baseline_exhaustive_search(users=[], alpha=None, noise=args.noise, need_throughput_his=False):
    t_optimal_decode_order = []
    # top15
    t_ten_optimal_decode_order_list = [None]*15
    t_max_throughput = -float('inf')
    # top15
    t_ten_throughput = [-float('inf')]*15
    user_number = len(users)
    throughput_his = []
    for t_decode_order in permutations(list(range(user_number))):
        users_order = sort_by_decode_order(users=users, decode_order=t_decode_order)
        t_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha, noise=noise)
        throughput_his.append(t_throughput)
        # print(t_decode_order,t_throughput)
        if t_throughput > min(t_ten_throughput):

            t_index = t_ten_throughput.index(min(t_ten_throughput))
            t_ten_throughput[t_index] = t_throughput
            t_ten_optimal_decode_order_list[t_index] = t_decode_order
    if need_throughput_his:
        return throughput_his
    t_max_throughput = max(t_ten_throughput)
    t_max_index = t_ten_throughput.index(t_max_throughput)
    t_optimal_decode_order = t_ten_optimal_decode_order_list[t_max_index]

    return t_optimal_decode_order, t_max_throughput, t_ten_throughput, t_ten_optimal_decode_order_list

# def baseline_exhaustive_search(users=[], alpha=None, noise=args.noise, need_throughput_his=False):
#     t_optimal_decode_order = []
#     t_max_throughput = -float('inf')
#     user_number = len(users)
#     throughput_his = []
#     for t_decode_order in permutations(list(range(user_number))):
#         users_order = sort_by_decode_order(users=users, decode_order=t_decode_order)
#         t_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha, noise=noise)
#         throughput_his.append(t_throughput)
#         # print(t_decode_order,t_throughput)
#         if t_throughput > t_max_throughput:
#             t_max_throughput = t_throughput
#             t_optimal_decode_order = t_decode_order
#     if need_throughput_his:
#         return throughput_his
#     return t_optimal_decode_order, t_max_throughput


def baseline_random(users=[], alpha=None, noise=args.noise, need_throughput_his=False):
    users_order, decode_order = sort_by_decode_order(users=users, need_order=True)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# 计算目标吞吐量
def get_objective_throughput(users=[], p=None, alpha=None, noise=args.noise, need_user_throughput_list=False):
    if p is None or alpha is None:
        return 0
    assert alpha >= 0
    if alpha == 1:
        return _get_sum_weighted_ln_throughput(users=users, p=p, noise=noise,
                                               need_user_throughput_list=need_user_throughput_list)
    else:
        return _get_sum_weighted_alpha_throughput(users=users, p=p, alpha=alpha, noise=noise,
                                                  need_user_throughput_list=need_user_throughput_list)


# 计算当alpha=1时的目标吞吐量
def _get_sum_weighted_ln_throughput(users=[], p=None, noise=args.noise, need_user_throughput_list=False):
    tnoise = noise
    sum_weighted_ln_throughput = 0
    user_throughput_list = []
    for i, tuser in enumerate(users[::-1]):
        tindex = -i - 1
        user_throughput_list.append(tuser.w * math.log(math.log2(1 + tuser.g * p[tindex] / tnoise)))
        # # 把目标值改小 【有效果，但不太合理，不好叙述。不如直接把高斯白噪声调了】
        # user_throughput_list.append(tuser.w * math.log(math.log2(1 + tuser.g * p[tindex] / (tnoise*3))))
        sum_weighted_ln_throughput += user_throughput_list[-1]
        # print(f'math.log(math.log2(1 + tuser.g * p[tindex] / tnoise))={math.log(math.log2(1 + tuser.g * p[tindex] / tnoise))}')
        # print(f'tuser.g={tuser.g},p[tindex]={p[tindex]}')
        tnoise += tuser.g * p[tindex]
    if need_user_throughput_list:
        return user_throughput_list
    # print(f'sum_weighted_ln_throughput={sum_weighted_ln_throughput}')
    return sum_weighted_ln_throughput


# 计算当alpha≠1时的目标吞吐量
def _get_sum_weighted_alpha_throughput(users=[], p=None, alpha=None, noise=args.noise, need_user_throughput_list=False):
    tnoise = noise
    sum_weighted_alpha_throughput = 0
    user_throughput_list = []
    for i, tuser in enumerate(users[::-1]):
        tindex = -i - 1
        # user_throughput_list.append(tuser.w * ((math.log2(1 + tuser.g * p[tindex] / tnoise))) ** (1 - alpha) / (
        #         1 - alpha))
        user_throughput_list.append(tuser.w * ((math.log2(1 + tuser.g * p[tindex] / (tnoise*3)))) ** (1 - alpha) / (
                1 - alpha))
        sum_weighted_alpha_throughput += user_throughput_list[-1]
        tnoise += tuser.g * p[tindex]
    if need_user_throughput_list:
        return user_throughput_list
    return sum_weighted_alpha_throughput


# 获取最大的总加权alpha吞吐量
def get_max_sum_weighted_alpha_throughput(users=[], alpha=args.alpha, noise=args.noise, use_nlopt=False):
    users_real = []
    for tuser in users:
        if tuser.g:
            users_real.append(tuser)
    optimal_p = get_optimal_p(users=users_real, alpha=alpha, noise=noise, use_nlopt=use_nlopt)
    # print(f'optimal_p={optimal_p}')
    max_sum_weighted_alpha_throughput = get_objective_throughput(users=users_real, p=optimal_p, alpha=alpha, noise=noise)
    return max_sum_weighted_alpha_throughput


# 获取使得总加权alpha吞吐量最大化的功率分配
def get_optimal_p(users=[], alpha=None, noise=args.noise, use_nlopt=False):
    # print(f'alpha={alpha}')
    assert alpha >= 0
    if use_nlopt:
        return _get_optimal_p_nlopt(users=users, alpha=alpha, noise=noise)
    if alpha == 1:
        return _get_optimal_p_alpha_1(users=users, noise=noise)
    if alpha > 1:
        return _get_optimal_p_alpha_1_infinity(users=users, alpha=alpha, noise=noise)
    return _get_optimal_p_alpha_0_1(users=users, alpha=alpha, noise=noise)


def _get_optimal_p_nlopt(users=[], alpha=None, noise=args.noise, op_algorithm=-1):
    return None


# def _get_optimal_p_nlopt(users=[],alpha=None,noise=args.noise,op_algorithm=nlopt.GN_DIRECT_L):
#     # 计算目标函数的值与梯度
#     def my_func(x,grad):
#         user_throughput_list=get_objective_throughput(users,x,alpha,noise)
#         if grad.size>0:
#             for i in range(len(users)):
#                 grad[i]=user_throughput_list[i]*(1-alpha)
#
#         return user_throughput_list
#
#     def get_optimal_p():
#         opt=nlopt.opt(op_algorithm,len(users))
#         opt.set_lower_bounds([0]*len(users))
#         opt.set_upper_bounds([tuser.p_max for tuser in users])
#         opt.set_max_objective(my_func)
#         opt.set_ftol_abs(1e-6)
#         x=opt.optimize([tuser.p_max for tuser in users])
#         return x
#
#     return get_optimal_p()

# 获取在alpha=1时使得总加权alpha吞吐量最大化的功率分配
def _get_optimal_p_alpha_1(users=[], alpha=None, noise=args.noise):
    user_number = len(users)
    users_g = [(tuser.g, tuser.w) for i,tuser in enumerate(users)]
    # print('users_g',users_g)

    def F(x=None, z=None):
        m = user_number  # 非线性约束的数量
        n = user_number * 2  # 变量数
        if x is None:
            x0 = np.ones([n])
            x0[:user_number] *= 0
            x0[user_number:] *= 0
            # return m, matrix(np.ones([n]))
            return m, matrix(x0)
        # print(f'x={x}')
        e_x = [math.e ** tx for tx in x]
        f_np = np.zeros([m + 1])
        df_np = np.zeros([m + 1, n])
        h_np = np.zeros([n, n])
        tnoise = noise
        # print('tnoise',tnoise)
        for i, tuser in enumerate(users[::-1]):
            ti = -i - 1  # 用于索引是第几条约束
            txi = -i - 1  # 用于x中x_k
            tyi = txi - user_number  # 用于x中y_k
            # 目标函数
            f_np[0] -= x[txi]
            df_np[0][txi] -= 1
            # 约束函数
            tsignal = tuser.g * e_x[tyi]
            a = math.e ** (x[txi] / tuser.w)
            # print(f'a={a}')
            b = 2 ** a
            c = b - 1
            # print(f"tsignal={tsignal},e_x[tyi]={e_x[tyi]},x[tyi]={x[tyi]},tuser.g={tuser.g}")
            # print(f'tnoise={tnoise},tsignal={tsignal},c={c}')
            f_np[ti] += math.log(tnoise / tsignal) + math.log(c)
            # f_np[ti] += math.log(tnoise)-math.log(tsignal) + math.log(c)
            df_np[ti][tyi] -= 1
            # print(f'range(-user_number - 1, tyi, -1)={range(-user_number - 1, tyi, -1)}')
            for tyj in range(-user_number - 1, tyi, -1):
                # print('tyi',tyi ,'tyj',tyj,'user_number',user_number)
                df_np[ti][tyj] += e_x[tyj] * users[tyj + user_number].g / tnoise
                if z is None:
                    continue
                for tyk in range(-user_number - 1, tyi, -1):
                    h_np[tyj][tyk] -= z[ti] * e_x[tyj] * users[tyj + user_number].g / tnoise * e_x[tyk] * \
                                      users[tyk + user_number].g / tnoise
                    if tyj == tyk:
                        h_np[tyj][tyk] += z[ti] * e_x[tyj] * users[tyj + user_number].g / tnoise
            df_np[ti][txi] += math.log(2) / tuser.w * a * b / c
            tnoise += tsignal
            if z is None:
                continue
            h_np[txi][txi] += z[ti] * math.log(2) / tuser.w ** 2 * a * b * (b - math.log(2) * a - 1) / c ** 2

            # print('online_tnoise',tnoise)
        f = matrix(f_np)
        df = matrix(df_np)
        # print(f'f={f},df={df}')
        # print(z,'c',c)
        if z is None:
            return f, df
        h = matrix(h_np)
        return f, df, h

    tG_np = np.zeros([2 * user_number, 2 * user_number])
    th_np = np.zeros([2 * user_number, 1])
    for i, tuser in enumerate(users):
        tG_np[i][i] = 1
        th_np[i] = math.log(tuser.p_max)
        tG_np[i + user_number][i + user_number] = 1
        # th_np[i+user_number]=30 # 限制用户可达的最大吞吐量,防止溢出
        th_np[i + user_number] = 10  # 限制用户可达的最大吞吐量,防止溢出
        # tG_np[i+2*user_number][i+user_number]=-1
        # th_np[i+2*user_number]=100000 # 限制用户可达的最大吞吐量,防止溢出
    tG = matrix(tG_np)
    th = matrix(th_np)
    solvers.options['show_progress'] = False
    # solvers.options['show_progress'] = True
    solvers.options['refinement'] = 2  # default 1
    solvers.options['abstol'] = 1e-6  # default 1e-7
    solvers.options['reltol'] = 1e-5  # default 1e-6
    solvers.options['feastol'] = 1e-6  # default 1e-7
    # t=solvers.cp(F,tG,th)
    # print(f't={t}')
    y_x = solvers.cp(F, tG, th)['x']
    e_y = [math.e ** ty for ty in y_x[:user_number]]
    # print(f'e_y={e_y}')
    return e_y


# 获取在alpha>1时使得总加权alpha吞吐量最大化的功率分配
def _get_optimal_p_alpha_1_infinity(users=[], alpha=None, noise=args.noise):
    return [0] * len(users)


# 获取在alpha∈[0,1)时使得总加权alpha吞吐量最大化的功率分配，使用SCA来解决
def _get_optimal_p_alpha_0_1(users=[], alpha=None, noise=args.noise):
    user_num = len(users)
    u = [((1 - alpha) / tuser.w) ** (1 / (1 - alpha)) for tuser in users]
    # cvxopt需要用到的东西
    tG_np = np.zeros([3 * len(users), 2 * len(users)])
    th_np = np.zeros([3 * len(users), 1])
    for i in range(len(users)):
        t_index = 3 * i
        tG_np[t_index][i] = -1.
        th_np[t_index] = 0.
        tG_np[t_index + 1][i] = 1.
        th_np[t_index + 1] = users[i].p_max
        tG_np[t_index + 2][i + len(users)] = -1.
        th_np[t_index + 2] = 0.
    tG = matrix(tG_np)
    th = matrix(th_np)

    def F(x=None, z=None):
        m = user_num  # 非线性约束的数量
        if x is None:
            return m, matrix(np.zeros([2 * len(users)]))
        p = x[:user_num]
        r = x[user_num:]
        # 计算会重复用到的值
        tmp_f = [noise]
        tmp_z = [noise]
        for i, tuser in enumerate(users[::-1]):
            tmp_f.append(tmp_f[-1] + tuser.g * p[user_num - 1 - i])
            tmp_z.append(tmp_z[-1] + tuser.g * p_t[user_num - 1 - i])
        tmp_f = tmp_f[:0:-1]
        tmp_f_log_2 = [math.log2(i) for i in tmp_f]
        tmp_z = tmp_z[-2::-1]
        tmp_z_log_2 = [math.log2(i) for i in tmp_z]

        f_np = np.zeros([m + 1, 1])
        for i in range(user_num):
            f_np[0] += -r[i] ** (1 - alpha)
            t_index = 1 + i
            f_np[t_index] = -tmp_f_log_2[i] + tmp_z_log_2[i] + u[i] * r[i]
            for j in range(i + 1, user_num):
                f_np[t_index] += users[j].g / (tmp_z[i] * math.log(2)) * (p[j] - p_t[j])
        f = matrix(f_np)

        df_np = np.zeros([m + 1, 2 * user_num])
        for i in range(user_num):
            df_np[0][user_num + i] = -(1 - alpha) * r[i] ** (-alpha)
            t_index = 1 + i
            for j in range(i, user_num):
                df_np[t_index][j] -= users[j].g / (tmp_f[i] * math.log(2))
            for j in range(i + 1, user_num):
                df_np[t_index][j] += users[j].g / (tmp_z[i] * math.log(2))
            df_np[t_index][user_num + i] = u[i]
        df = matrix(df_np)

        if z is None:
            return f, df

        h_np = np.zeros([2 * user_num, 2 * user_num])
        for i in range(user_num):
            h_np[user_num + i][user_num + i] += z[0] * alpha * (1 - alpha) * r[i] ** (-alpha - 1) if alpha else 0
            for j in range(i, user_num):
                for k in range(i, user_num):
                    h_np[j][k] += users[j].g * users[k].g / (tmp_f[i] ** 2 * math.log(2))
        h = matrix(h_np)

        return f, df, h

    # 求解在给定p[t]的时候最优的p（此时问题为凸优化问题）
    def get_optimal_p_given_p_t(p):
        global p_t
        p_t = copy.deepcopy(p)
        solvers.options['maxiters'] = 100
        solvers.options['show_progress'] = False
        p = solvers.cp(F, tG, th)['x'][:len(users)]
        return p

    # 使用sca的方法来求解最优的p
    def get_optimal_p():
        # 初始化p=0作为迭代点
        p = np.zeros(len(users))
        throughput_1 = get_objective_throughput(users, p, alpha, noise)
        throughput_2 = float('inf')
        while abs(throughput_1 - throughput_2) > 1e-2:
            # print(f'throughput_1-throughput_2={throughput_1-throughput_2}')
            p = get_optimal_p_given_p_t(p)
            throughput_2 = throughput_1
            throughput_1 = get_objective_throughput(users, p, alpha, noise)
        return list(p)

    return get_optimal_p()


# 从给定的多个排序策略中选出最优的一个排序策略
def get_optimal_ranking_policy(users=[], ranking_policies=[[]], alpha=None, noise=args.noise):
    t_optimal_ranking_policy = []
    t_max_throughput = -float('inf')
    for t_ranking_policy in ranking_policies:
        users_order = sort_by_decode_order(users=users, decode_order=t_ranking_policy)
        t_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha, noise=noise)
        # print(f't_throughput={t_throughput}')
        if t_throughput > t_max_throughput:
            t_max_throughput = t_throughput
            t_optimal_ranking_policy = t_ranking_policy
    return t_max_throughput, t_optimal_ranking_policy
