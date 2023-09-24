import argparse

parser=argparse.ArgumentParser()
# 用来生成拓扑的
parser.add_argument('--user_num',default=5,help='用户数')
# 1,2,1,10
parser.add_argument('--d_min',default=20,help='用户到基站的最小距离, default 20')
parser.add_argument('--d_max',default=100,help='用户到基站的最大距离, default 100')
parser.add_argument('--w_min',default=1,help='用户的最小吞吐量权重') #1,5
parser.add_argument('--w_max',default=1,help='用户最大的吞吐量权重')
# 与问题相关的参数
parser.add_argument('--alpha',default=1,help='alpha公平性中的alpha,当alpha=1时表示对数公平性')
# 9
# parser.add_argument('--noise',default=3.981e-17,help='基站的高斯白噪声,default -184 dBm/Hz * 1MHz, i.e., 3.981e-16') #1e-8~1e-13
# parser.add_argument('--noise',default=3.981e-17,help='基站的高斯白噪声,default -184 dBm/Hz * 1MHz, i.e., 3.981e-16') #1e-8~1e-13
parser.add_argument('--noise',default=3.981e-15,help='基站的高斯白噪声,default -184 dBm/Hz * 1MHz, i.e., 3.981e-16') #1e-8~1e-13
# 随机化相关
parser.add_argument('--seed',default=1234,help='随机数种子')
# 与学习相关的
parser.add_argument('--frames_num',default=300000,help='时间帧的帧数')
# 与数据集相关
parser.add_argument('--train_data_num',default=120000,help='用来训练的数据量')
# parser.add_argument('--train_data_num',default=120,help='用来训练的数据量')
parser.add_argument('--val_data_num',default=100,help='用来验证的数据量')
parser.add_argument('--train_batch_size',default=1024,help='用来训练的批次大小')
parser.add_argument('--val_batch_size',default=1024,help='用来验证的批次大小')
# 与pointer_network有关
parser.add_argument('--embedding_dim',default=2,help='pointer network中对输入的embedding的维度')
parser.add_argument('--hidden_dim',default=2,help='pointer network中隐藏层的维度')
parser.add_argument('--use_cuda',default=False,help='是否使用gpu')
# 与训练有关
parser.add_argument('--epoch_num',default=2,help='训练的周期数')
parser.add_argument('--lr',default=1e-3,help='学习速率')
parser.add_argument('--reward_beta',default=0.9,help='控制训练时平均reward的更新速度,越大更新速度越慢')



args=parser.parse_args()