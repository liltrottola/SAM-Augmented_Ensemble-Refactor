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


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        # Initialize the AvgMeter with a default window size of 40
        # This window size determines how many of the most recent loss values will be averaged
        self.num = num
        # Call the reset method to initialize the statistics
        self.reset()

    def reset(self):
        # Reset all internal statistics
        # This is useful when starting a new evaluation or epoch
        self.val = 0  # The current value of the metric
        self.avg = 0  # The average value of the metric
        self.sum = 0  # The cumulative sum of the metric values
        self.count = 0  # The number of values added
        self.losses = []  # A list to keep track of all individual loss values

    def update(self, val, n=1):
        # Update the AvgMeter with a new value
        # 'val' is the new value to add, and 'n' is the number of occurrences of this value
        self.val = val  # Update the current value to the new value
        self.sum += val * n  # Add the new value to the cumulative sum, considering its frequency
        self.count += n  # Increment the count by the number of occurrences
        self.avg = self.sum / self.count  # Recalculate the average
        self.losses.append(val)  # Append the new value to the list of losses

    def show(self):
        # Return the average of the most recent 'num' values
        # Compute the average of the last 'num' losses from the list
        # If there are fewer than 'num' losses, average over all available losses
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
