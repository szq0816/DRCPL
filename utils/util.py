import os
import sys
import math
import errno
import shutil
import random

import torch
import numpy as np


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        print(f"=> loading models '{model_file}' epoch:{checkpoint['epoch']} map_all:{checkpoint['map_all']:.3f}")
        return checkpoint
    else:
        print("=> no models found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def adjust_learning_rate(args, optimizer, epoch):
    # lr = args.LR * (args.LR_GAMMA ** (float(epoch) / args.MAX_EPOCH))
    lr = args.MIN_LR + 0.5 * (args.LR - args.MIN_LR) * (1 + math.cos(epoch / (args.MAX_EPOCH - 1) * math.pi))
    print(f'epoch: {epoch + 1}, lr: {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_dir(root_save_path):
    if os.path.exists(root_save_path):
        shutil.rmtree(root_save_path)  # delete output folder
    os.makedirs(root_save_path)  # make new output folder


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(m):
    total_param = sum(p.numel() for p in m.parameters())
    trainable_param = sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("total_param: ", total_param)
    print("trainable_param: ", trainable_param)

    # return total_param, trainable_param
