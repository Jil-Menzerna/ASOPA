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


# 生成拓扑，返回一个用户的list
def generate_topology(user_number=args.user_num, d_min=args.d_min, d_max=args.d_max, w_min=args.w_min,
                      w_max=args.w_max, ):
    # 暂时假设用户在[d_min,d_max]间均匀分布
    d = [(td+1) * (d_max - d_min) / (user_number) + d_min for td in range(user_number)]
    # 令各用户到基站的距离在d_min到d_max之间随机
    # d = np.random.random([user_number]) * (d_max - d_min) + d_min
    # 计算各用户的上行平均信道增益，暂时与TMC的计算公式相同
    g = [4.11 * (3e8 / (4 * math.pi * 915e6 * td)) ** 2.8 for td in d]   # g_hat相当于，只与距离有关
    print(f'\ng={g}')
    # 暂时令各用户的权重为w_min与w_max交替的形式
    w = np.ones(user_number)
    # w = [w_min if i % 2 == 0 else w_max for i, _ in enumerate(w)]
    # 令各用户的权重在w_min与w_max之间随机
    # w=np.random.random([user_number])*(w_max-w_min)+w_min
    # 令用户的权重在[w_min, w_max]之间均匀分布,从小到大
    # w = [w_min + i * (w_max - w_min) / (user_number - 1) for i in range(user_number)]
    # 从大到小
    # w = w[::-1]
    # 令用户的权重从给定的列表中随机选择
    # w_possible = [1, 4, 16]
    w_possible = [1, 2, 4, 8, 16, 32] # 现在的基准是这个

    # w_possible = [1, 32] # 把基准除掉一些
    # w_possible = [1, 2, 4, 8, 16, 32] # 换成10看看显著不显著

    # w_possible = [1, 2, 4, 8]
    # w_possible = [1, 2 ]
    w_index = np.random.randint(0, len(w_possible), size=user_number)
    w = [w_possible[index] for index in w_index]
    # # 非随机测试
    # w = [1, 3, 3, 1]
    # 让用户的最大发射功率在1到3之间
    # p_max = np.random.random(user_number) * 2 + 1
    # 令用户的发送功率从给定的列表中随机选择
    # w_possible = [1, 4, 16, 64]
    # p_max_possible = [1,2]
    p_max_possible = [1]
    p_max_index = np.random.randint(0, len(p_max_possible), size=user_number)
    p_max = [p_max_possible[index] for index in p_max_index]
    # 让用户的最大发送功率为[1,2,1,2,1,2.....]
    # p_max = [1 if i%2==0 else 2 for i in range(user_number)]
    print(f'd={d}')
    # random.shuffle(w)
    print(f'w={w}')
    # random.shuffle(p_max)
    print(f'p_max={p_max}')
    # 创建用户数组
    users = []
    for i in range(user_number):
        users.append(User(tid=i, tp_max=p_max[i], tg=g[i], tw=w[i], td=d[i]))
    # random.shuffle(users)
    # users=users[::-1]
    return users

def H_generator(users,noise,i):
    H_matrix = [[], []]
    for j, fuser in enumerate(users[::-1]):
        if j <= i:
            H_matrix[0].append(fuser.g[0] + fuser.g[1] * 1j)
            H_matrix[1].append(fuser.g[2] + fuser.g[3] * 1j)

    # print('F',H_matrix)
    H_matrix = np.matrix(H_matrix)
    T_MMSE = np.linalg.inv((H_matrix.H @ H_matrix) + np.sqrt(noise) * np.eye(i + 1)) @ H_matrix.H
    H_2 = T_MMSE @ H_matrix
    MIMO_g = []
    for k_index in range(i + 1):
        # print('i,k_index',i,k_index)
        # print(H_2)
        MIMO_g.append(np.linalg.norm(H_2[i,k_index])*np.linalg.norm(H_2[i,k_index]))
    MIMO_g = np.flip(MIMO_g)
    return MIMO_g,T_MMSE

def H_generator_duibi(users,noise,i):
    H_matrix = [[], []]
    for j, fuser in enumerate(users[::-1]):     # 从最后一位高斯开始取，倒过来了
        H_matrix[0].append(fuser.g[0] + fuser.g[1] * 1j)
        H_matrix[1].append(fuser.g[2] + fuser.g[3] * 1j)

    # print('F',H_matrix)
    H_matrix = np.matrix(H_matrix)
    T_MMSE = np.linalg.inv((H_matrix.H @ H_matrix) + np.sqrt(noise) * np.eye(i)) @ H_matrix.H
    H_2 = T_MMSE @ H_matrix
    MIMO_g = np.zeros([args.user_num,args.user_num])
    # print(len(H_2),len(H_2[:,]))
    for k_index in range(len(H_2)):
        for j_index in range(len(H_2[:,])):
            MIMO_g[k_index,j_index] = np.linalg.norm(H_2[k_index, j_index])*np.linalg.norm(H_2[k_index, j_index])
    # print(MIMO_g)
    return MIMO_g,T_MMSE

# 根据解码顺序对用户进行排序,如果不给定解码顺序就随机排序
def sort_by_decode_order(users=[], decode_order=None, need_order=False):
    if decode_order is None:
        decode_order = list(range(len(users)))
        random.shuffle(decode_order)
    users_order = [users[i] for i in decode_order]
    if need_order:
        return users_order, decode_order
    return users_order


# 所有对比方法都返回其对应的解码顺序，与在这个解码顺序下的最大的总加权alpha吞吐量
# 对比方案，按照g的升序
def duibi_g_order_asc(users=[], alpha=None, noise=args.noise):

    a = [(i, tuser.g) for i, tuser in enumerate(users)]
    H_mal,T_MMSE = H_generator_duibi(users,noise,len(a))
    H_snr = []
    # print('H_mal',H_mal)
    for i in range(len(H_mal)):
        H_noise = noise* np.linalg.norm(T_MMSE[i])**2
        for j in range(len(H_mal)):
            if j != i:
                H_noise += H_mal[i,j]
        H_snr.append((len(a)-i-1, H_mal[i,i]/(H_noise)))
    print('H_snr',H_snr)
    index_sort = sorted(H_snr, key=lambda ta: ta[1])
    decode_order = [i for i, _ in index_sort]
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput

# 对比方案，按照g的降序
# def duibi_g_order_desc(users=[], alpha=None, noise=args.noise):
#
#     a = [(i, tuser.g) for i, tuser in enumerate(users)]
#     H_mal,T_MMSE = H_generator_duibi(users,noise,len(a))
#     H_snr = []
#     # print('H_mal',H_mal)
#     for i in range(len(H_mal)):
#         H_noise = noise* np.linalg.norm(T_MMSE[i])**2
#         for j in range(len(H_mal)):
#             if j == i:
#                 continue
#
#             H_noise += H_mal[i,j]
#         H_snr.append((len(a)-i-1, H_mal[i,i]/(H_noise)))
#     print(H_snr,type(H_snr))
#     index_sort = sorted(H_snr, key=lambda ta: -ta[1])
#     decode_order = [i for i, _ in index_sort]
#     users_order = sort_by_decode_order(users=users, decode_order=decode_order)
#     max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
#                                                                               noise=noise)
#     return decode_order, max_sum_weighted_alpha_throughput

# 不用信噪比，直接用Hk排序
def duibi_g_order_desc(users=[], alpha=None, noise=args.noise):
    H_k = []
    for j, fuser in enumerate(users[::-1]):     # 从最后一位高斯开始取，倒过来了
        H_k.append((len(users)-j-1,np.linalg.norm([fuser.g[0] + fuser.g[1] * 1j,fuser.g[2] + fuser.g[3] * 1j])))
        # print('j',j)
    print('H_k',H_k)

    index_sort = sorted(H_k, key=lambda ta: -ta[1])
    decode_order = [i for i, _ in index_sort]
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput
# # 对比方案，按照g的降序
# def duibi_g_order_desc(users=[], alpha=None, noise=args.noise):
#     a = [(i, tuser.g) for i, tuser in enumerate(users)]
#     a = sorted(a, key=lambda ta: -ta[1])
#     decode_order = [i for i, _ in a]
#     # print(f'{decode_order}')
#     users_order = sort_by_decode_order(users=users, decode_order=decode_order)
#     max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
#                                                                               noise=noise)
#     return decode_order, max_sum_weighted_alpha_throughput


# 对比方案，按照用户权重w的降序
def duibi_w_order_desc(users=[], alpha=None, noise=args.noise):
    a = [(i, tuser.w) for i, tuser in enumerate(users)]
    a = sorted(a, key=lambda ta: -ta[1])
    decode_order = [i for i, _ in a]
    # print(f'{decode_order}')
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# 对比方案，按照用户权重w的升序
def duibi_w_order_aesc(users=[], alpha=None, noise=args.noise):
    a = [(i, tuser.w) for i, tuser in enumerate(users)]
    a = sorted(a, key=lambda ta: ta[1])
    decode_order = [i for i, _ in a]
    # print(f'{decode_order}')
    users_order = sort_by_decode_order(users=users, decode_order=decode_order)
    max_sum_weighted_alpha_throughput = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha,
                                                                              noise=noise)
    return decode_order, max_sum_weighted_alpha_throughput


# 对比方案，qian的启发式方法（不断找到个最好的位置插入）（默认按照给定的用户顺序进行插入）
def duibi_heuristic_method_qian(users=[], alpha=None, noise=args.noise, need_random=False):
    # print('用户状态',users)
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


# 对比方法，qian的启发式方法（不断找到个最好的位置插入）（先打乱用户，然后按照打乱之后的用户顺序进行插入）
def duibi_heuristic_method_qian_random(users=[], alpha=None, noise=args.noise):
    return duibi_heuristic_method_qian(users,alpha,noise,need_random=True)

# 对比方案，穷搜
def duibi_exhaustive_search(users=[], alpha=None, noise=args.noise, need_throughput_his=False):
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

# def duibi_exhaustive_search(users=[], alpha=None, noise=args.noise, need_throughput_his=False):
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


def duibi_random(users=[], alpha=None, noise=args.noise, need_throughput_his=False):
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

    sum_weighted_ln_throughput = 0
    user_throughput_list = []
    for i, tuser in enumerate(users[::-1]):
        tindex = -i - 1
        MIMO_g,T_MMSE = H_generator(users, noise, i)
        # print('目标MIMO_g',MIMO_g,i)
        # tnoise = noise* np.linalg.norm(T_MMSE[i])**2        # 服了! 这么错是吧
        tnoise = noise* np.linalg.norm(T_MMSE[tindex])**2
        # 每次重新计算干扰
        for tyj in range(-1, -i - 1, -1):
            tnoise += p[tyj] * MIMO_g[tyj]

        user_throughput_list.append(tuser.w * math.log(math.log2(1 + MIMO_g[tindex] * p[tindex] / tnoise)))
        # # 把目标值改小 【有效果，但不太合理，不好叙述。不如直接把高斯白噪声调了】
        # user_throughput_list.append(tuser.w * math.log(math.log2(1 + tuser.g * p[tindex] / (tnoise*3))))
        sum_weighted_ln_throughput += user_throughput_list[-1]
        # print(f'math.log(math.log2(1 + tuser.g * p[tindex] / tnoise))={math.log(math.log2(1 + tuser.g * p[tindex] / tnoise))}')
        # print(f'tuser.g={tuser.g},p[tindex]={p[tindex]}')

    if need_user_throughput_list:
        return user_throughput_list
    # print(f'sum_weighted_ln_throughput={sum_weighted_ln_throughput}')
    return sum_weighted_ln_throughput


# 计算当alpha≠1时的目标吞吐量
def _get_sum_weighted_alpha_throughput(users=[], p=None, alpha=None, noise=args.noise, need_user_throughput_list=False):

    sum_weighted_alpha_throughput = 0
    user_throughput_list = []
    for i, tuser in enumerate(users[::-1]):
        tindex = -i - 1
        MIMO_g,T_MMSE = H_generator(users,noise,i)
        tnoise = noise* np.linalg.norm(T_MMSE[i])**2
        for tyj in range(-1, -i-1, -1):
            tnoise += p[tyj] * MIMO_g[tyj]

        # user_throughput_list.append(tuser.w * ((math.log2(1 + tuser.g * p[tindex] / tnoise))) ** (1 - alpha) / (
        #         1 - alpha))
        user_throughput_list.append(tuser.w * ((math.log2(1 + MIMO_g[tindex] * p[tindex] / (tnoise*3)))) ** (1 - alpha) / (
                1 - alpha))
        sum_weighted_alpha_throughput += user_throughput_list[-1]

    if need_user_throughput_list:
        return user_throughput_list
    return sum_weighted_alpha_throughput


# 获取最大的总加权alpha吞吐量
def get_max_sum_weighted_alpha_throughput(users=[], alpha=args.alpha, noise=args.noise, use_nlopt=False):
    optimal_p,throughput_ = get_optimal_p(users=users, alpha=alpha, noise=noise, use_nlopt=use_nlopt)
    # print(f'optimal_p={optimal_p}',f'throughput_={throughput_}')
    max_sum_weighted_alpha_throughput = get_objective_throughput(users=users, p=optimal_p, alpha=alpha, noise=noise)
    # print('max_sum_throughput',max_sum_weighted_alpha_throughput)
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
    # print('users',users,len(users))

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
        e_x = [math.e ** tx for tx in x]        # e_x ---> e^yn
        f_np = np.zeros([m + 1])                # m+1个非线性约束
        df_np = np.zeros([m + 1, n])            # (m+1)非线性约束，对n个变量的一阶导
        h_np = np.zeros([n, n])                 #
        T_MMSE = []

        # i=3
        # H_matrix = [[], []]
        # for j, tuser in enumerate(users[::-1]):
        #     if j <= i:
        #         H_matrix[0].append(tuser.g[0] + tuser.g[1] * 1j)
        #         H_matrix[1].append(tuser.g[2] + tuser.g[3] * 1j)
        #
        # H_matrix = np.matrix(H_matrix)
        # T_MMSE = np.linalg.inv((H_matrix.H @ H_matrix) + noise * np.eye(i + 1)) @ H_matrix.H
        # H_2 = T_MMSE @ H_matrix
        # MIMO_g = []
        # for k_index in range(i + 1):
        #     MIMO_g.append(np.linalg.norm(H_2[i, k_index]))
        # print('MIMO_g', MIMO_g)

        MIMO_g =[[] for i in range(10)]
        for i, tuser in enumerate(users[::-1]):         # 从最后一个解码的开始取，因为最后一位解码的只有高斯白噪声
            # 获得最大的信号矩阵 # H_matrix就是倒序建立的 所以后面要flip
            H_matrix = [[], []]
            for j, fuser in enumerate(users[::-1]):
                if j <= i:
                    H_matrix[0].append(fuser.g[0] + fuser.g[1] * 1j)
                    H_matrix[1].append(fuser.g[2] + fuser.g[3] * 1j)

            H_matrix = np.matrix(H_matrix)
            T_MMSE = np.linalg.inv((H_matrix.H @ H_matrix) + np.sqrt(noise) * np.eye(i + 1)) @ H_matrix.H  # 为这里计算的是幅值的T_MMSE 所以高斯白噪声功率应该要开个根号
            H_2 = T_MMSE @ H_matrix
            # print('T_MMSE',T_MMSE)
            # MIMO_g_,T_MMSE_ = H_generator(users,noise,i)
            # print('MIMO_g_',MIMO_g_,'T_MMSE_',T_MMSE_)
            # MIMO_g[i] = []
            for k_index in range(i+1):
                # print(np.linalg.norm(H_2[i,k_index]))
                # print(np.linalg.norm(H_2[i,k_index])*np.linalg.norm(H_2[i,k_index]))
                MIMO_g[i].append(np.linalg.norm(H_2[i,k_index])*np.linalg.norm(H_2[i,k_index]))
            MIMO_g[i] = np.flip(MIMO_g[i]) # 翻转数组
            # print('MIMO_g',MIMO_g[i],'T_MMSE',T_MMSE)
            # print('求解MIMO_g', MIMO_g, i)
            # print('MIMO_g',MIMO_g)
            # print('MIMO_g i -1',MIMO_g[i],MIMO_g[-1])
            # print('i,tuser',i,tuser)
            # print('tuser.g',tuser.g)
            # print('H',tuser.H)
            ti = -i - 1  # 用于索引是第几条约束
            txi = -i - 1  # 用于x中x_k             即v_n 替换后的吞吐量变量      -i-1    后N个变量的最后的元素
            tyi = txi - user_number  # 用于x中y_k  即y_n e^{y_n} =x_n           -i-1-user_numer  前N个变量的最后元素
            # 目标函数
            f_np[0] -= x[txi]                       # 求解吞吐量的负值，即最小值 x[txi]-->vn?
            df_np[0][txi] -= 1                      # min -sum(vn) 因此一阶导为-1
            # 约束函数
            tsignal = MIMO_g[i][txi] * e_x[tyi]            # tsignal = g*e**y_n    约束2中的信道增益乘发射功率
            a = math.e ** (x[txi] / tuser.w)        # e**(v_n/w_n)
            # print(f'a={a}')
            b = 2 ** a                              # 2**(e**(v_n/w_n))
            c = b - 1                               # 2**(e**(v_n/w_n)) -1 约束2的变形
            # print(f"tsignal={tsignal},e_x[tyi]={e_x[tyi]},x[tyi]={x[tyi]},tuser.g={tuser.g}")
            # print(f'tnoise={tnoise},tsignal={tsignal},c={c}')

            # 每次噪声都从高斯白噪声开始叠加
            tnoise = noise * np.linalg.norm(T_MMSE[txi])**2 # 高斯白噪声也受T_MMSE影响
            # print('tnoise',tnoise)  # 这里有这么大？

            for tyj in range(-user_number - 1, tyi, -1):
                tnoise += e_x[tyj] * MIMO_g[i][tyj + user_number]

            f_np[ti] += math.log(tnoise / tsignal) + math.log(c)        # 目标函数
            # f_np[ti] += math.log(tnoise)-math.log(tsignal) + math.log(c)
            df_np[ti][tyi] -= 1    # 当前解码用户的导数为-1？
            # print(f'range(-user_number - 1, tyi, -1)={range(-user_number - 1, tyi, -1)}')


            for tyj in range(-user_number - 1, tyi, -1):
                df_np[ti][tyj] += e_x[tyj] * MIMO_g[i][tyj + user_number] / tnoise  # tnoise全都得改 改成随循环变动得 这个一阶导就很怪... 是做过转换后的一阶导。
                if z is None:
                    continue
                for tyk in range(-user_number - 1, tyi, -1):
                    h_np[tyj][tyk] -= z[ti] * e_x[tyj] * MIMO_g[i][tyj + user_number] / tnoise * e_x[tyk] * \
                                      MIMO_g[i][tyk + user_number] / tnoise
                    if tyj == tyk:
                        h_np[tyj][tyk] += z[ti] * e_x[tyj] * MIMO_g[i][tyj + user_number] / tnoise
            df_np[ti][txi] += math.log(2) / tuser.w * a * b / c

            if z is None:
                continue
            h_np[txi][txi] += z[ti] * math.log(2) / tuser.w ** 2 * a * b * (b - math.log(2) * a - 1) / c ** 2
        f = matrix(f_np)
        df = matrix(df_np)
        # print(f'f={f},df={df}')
        if z is None:
            return f, df
        h = matrix(h_np)
        return f, df, h

    tG_np = np.zeros([2 * user_number, 2 * user_number])
    th_np = np.zeros([2 * user_number, 1])
    for i, tuser in enumerate(users):
        # tG th 分别为 tG * x <= th 两边的矩阵
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
    throughput = sum(y_x[user_number:])
    # print(f'e_y={e_y}')
    # print('throughput',throughput)
    return e_y,throughput


# 获取在alpha>1时使得总加权alpha吞吐量最大化的功率分配
def _get_optimal_p_alpha_1_infinity(users=[], alpha=None, noise=args.noise):
    return [0] * len(users)


# 获取在alpha∈[0,1)时使得总加权alpha吞吐量最大化的功率分配，使用SCA来解决
def _get_optimal_p_alpha_0_1(users=[], alpha=None, noise=args.noise):
    user_num = len(users)
    u = [((1 - alpha) / tuser.w) ** (1 / (1 - alpha)) for tuser in users]
    # cvxopt需要用到的东西
    # p>=0  p<=pmax
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
