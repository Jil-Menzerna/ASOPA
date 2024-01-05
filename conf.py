import argparse

parser=argparse.ArgumentParser()
# generate topology
parser.add_argument('--user_num',default=10,help='The number of users')
parser.add_argument('--val_user_num',default=8,help='The size of the problem graph for validating')
# 1,2,1,10
parser.add_argument('--d_min',default=20,help='the lower limits of distance, default 20')
parser.add_argument('--d_max',default=100,help='the upper limits of distance, default 100')
parser.add_argument('--w_min',default=1,help='The minimum throughput weight of user') #1,5
parser.add_argument('--w_max',default=1,help='The maximum throughput weight of user')

parser.add_argument('--num_min',default=5,help='The minimum number of users') #1,5
parser.add_argument('--num_max',default=10,help='The maximum number of users')


# the parameter of problem
parser.add_argument('--alpha',default=1,help='alpha fairness, and alpha =1 denotes the logarithmic fairness')
# 9

parser.add_argument('--noise',default=3.981e-15,help='Gaussian white noise in BS, default -184 dBm/Hz * 1MHz, i.e., 3.981e-16') #1e-8~1e-13
# random seed
parser.add_argument('--seed',default=1234,help='random seed')
# learning
parser.add_argument('--frames_num',default=300000,help='/')
# dataset
parser.add_argument('--train_data_num',default=120000,help='/')
# parser.add_argument('--train_data_num',default=120,help='用来训练的数据量')
parser.add_argument('--val_data_num',default=1000,help='The size of validation dataset')
parser.add_argument('--train_batch_size',default=1024,help='/')
parser.add_argument('--val_batch_size',default=1024,help='/')
# network
parser.add_argument('--embedding_dim',default=2,help='/')
parser.add_argument('--hidden_dim',default=2,help='/')
parser.add_argument('--use_cuda',default=False,help='whether to use gpu')
# training
parser.add_argument('--epoch_num',default=2,help='/')
parser.add_argument('--lr',default=1e-3,help='learning rate')
parser.add_argument('--reward_beta',default=0.9,help='the update speed of average reward')



args=parser.parse_args()