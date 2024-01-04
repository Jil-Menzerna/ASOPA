import os
import json
import pprint as pp
import time

import torch
import torch.optim as optim

import numpy as np

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from show import show_speed_performance_dataset


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Figure out what's the problem
    problem =load_problem(opts.problem)


    model= torch.load('Variable_user.pth')
    # num_list = [10,12,14,16,18,20]
    time_list = []

    val_dataset = problem.load_val_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    opts.eval_batch_size = 1
    time_start = time.time()

    show_speed_performance_dataset(val_dataset)

    time_end = time.time()
    time_list.append(time_end-time_start)
    print('time_cost:',time_end-time_start)

if __name__ == "__main__":
    run(get_options())