# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import random
from utils import build_dataset, build_iterator, get_time_dif
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    time_start = time.time()
    dataset = 'hsk_all'  # 数据集
    model_name = 'bert'  # bert
    # linguistic_where = 'self-attention' # bert-embedding or self-attention
    # level = 'character-word-grammar'
    print('---Text Difficulty Classification for ' + dataset + '---')
    print('---Using ' + model_name + '---')
    x = import_module(model_name)
    config = x.Config(dataset)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    time_end = time.time()
    time_sum = time_end - time_start
    print('Finish! Using %.2fs' % time_sum)