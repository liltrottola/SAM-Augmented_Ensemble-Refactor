import torch
import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


'''
# original HGM adjust_lr: with epoch 0..99 and decay_epoch=50 it decays before epochs 50 and 100 (1-based),
# same as HSNet-style milestones [50, 100] with decay_factor 0.1 (see configs/hgm_vanilla.yaml)
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if (epoch+1) % decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
'''

def adjust_lr(optimizer, decay_rate=0.1):
    # same as HSNet utils: decay applied at the milestones checked in Train_vanilla.py
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))