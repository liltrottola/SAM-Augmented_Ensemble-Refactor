import random
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from SAM2RFANet_n3 import SAM2RFANet_N3                # Wrapper appena creato
from utils.dataloader_n3 import get_loader_n3      
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  
from src.training.lr_schedules import get_lr_method, build_scheduler
from src.augmentation import OFFLINE_DA_METHODS
from src.sweep import registry as reg   

import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config(object):
    def __init__(self, d):
        for k , v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def build_prior_dirs(opt, channels, dataset, da=None):
    prior_dirs = []
    for channel in channels:
        folder, prior = channel.split("/")
        if da is None:
            prior_dirs.append(os.path.join(opt.paths.prior_root, folder, dataset, prior))
        else:
            prior_dirs.append(os.path.join(opt.paths.prior_root, folder, da, dataset, prior))
    return prior_dirs

def train(train_loader, model, optimizer, epoch, opt, debug=False):
    model.train()
    size_rates = opt.training.size_rates
    clip_margin = opt.training.clip_margin

    loss_record = AvgMeter()
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        if debug and i > 5:
            break

        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            trainsize = int(round(opt.training.trainsize * rate / 32) * 32)

            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                
            pred0, pred1, pred2, pred3 = model(images)  

            loss0 = structure_loss(pred0, gts)
            loss1 = structure_loss(pred1, gts)
            loss2 = structure_loss(pred2, gts)
            loss3 = structure_loss(pred3, gts)
            loss = loss0 + loss1 + loss2 + loss3
            
            loss.backward()
            if clip_margin is not None:
                clip_gradient(optimizer, clip_margin)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.training.batchsize)
                
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}] lr'.format(
                      datetime.now(), epoch, opt.training.epochs, i, total_step,
                      loss_record.show()), optimizer.param_groups[-1]['lr']) 
                      
    save_path = opt.paths.models_dir
    os.makedirs(save_path, exist_ok=True)
    filename = f"{opt.model_name}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=str, default="../../../configs/sam2rfanet_n3.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr_method', type=str, default=None)
    parser.add_argument('--offline_da', type=str, default=None)
    parser.add_argument('--online_da', type=str, default=None)
    parser.add_argument('--train_dataset', type=str, default=None)
    
    parser.add_argument('--sam_priors', type=str, nargs='+', default=None)
    parser.add_argument('--depth_priors', type=str, nargs='+', default=None)
    parser.add_argument('--dsine_priors', type=str, nargs='+', default=None)
    parser.add_argument('--registry', type=str, default='../../../configs/sweep_registry.yaml')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=None)
    parser.add_argument('--reduce_lr_mult', type=float, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.config): exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data) 

    model_name = args.model_name if args.model_name is not None else opt.experiment.name
    opt.model_name = model_name  
    if args.seed is not None: opt.experiment.seed = args.seed
    if args.lr_method is not None: opt.training.lr_method = args.lr_method
    if args.offline_da is not None: opt.training.offline_augmentation = args.offline_da
    if args.online_da is not None: opt.training.online_augmentation = args.online_da
    if args.train_dataset is not None: opt.datasets.train = [args.train_dataset]

    registry = reg.load_registry(args.registry)
    cli_priors = {"sam": args.sam_priors, "depth": args.depth_priors, "dsine": args.dsine_priors}
    channels = reg.get_n3_priors(registry, cfg_data['n3'], cli_priors)
    residual = True if args.residual else getattr(opt.n3, "residual", False)
    kernel_size = args.kernel_size if args.kernel_size is not None else getattr(opt.n3, "kernel_size", 1)
    reduce_lr_mult = args.reduce_lr_mult if args.reduce_lr_mult is not None else getattr(opt.n3, "reduce_lr_mult", 1)
    in_channels = 3 + len(channels)

    logging.basicConfig(filename=opt.paths.log_file,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    if hasattr(opt.experiment, 'seed'):
        seed = opt.experiment.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # N3 Model initialization
    model = SAM2RFANet_N3(in_channels, residual, kernel_size, checkpoint_path=opt.paths.hiera_path).cuda()

    lr_cfg = get_lr_method(getattr(opt.training, "lr_method", None))
    init_lr = lr_cfg["init_lr"] if lr_cfg is not None else opt.training.optimizer.lr

    # Split param groups: ChannelReduction params e SAM2RFANet params
    params = [
        {"params": model.reduce.parameters(), "lr": init_lr * reduce_lr_mult},
        {"params": model.sam2rfanet.parameters()},   
    ]

    if opt.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params, init_lr, weight_decay=opt.training.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, init_lr, weight_decay=1e-4, momentum=0.9)

    if lr_cfg is not None:
        scheduler = build_scheduler(optimizer, lr_cfg)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.training.epochs, eta_min=opt.training.lr_schedule.eta_min)

    offline_aug = getattr(opt.training, "offline_augmentation", None)
    online_aug = getattr(opt.training, "online_augmentation", None)
    dataset = opt.datasets.train[0]

    if offline_aug in OFFLINE_DA_METHODS:
        image_root = '{}/{}/{}/images/'.format(opt.paths.offline_aug_base, offline_aug, dataset)
        gt_root    = '{}/{}/{}/masks/'.format(opt.paths.offline_aug_base, offline_aug, dataset)
        prior_dirs = build_prior_dirs(opt, channels, dataset, da=offline_aug)
    else:
        image_root = '{}/{}/images/'.format(opt.paths.datasets_root, dataset)
        gt_root    = '{}/{}/masks/'.format(opt.paths.datasets_root, dataset)
        prior_dirs = build_prior_dirs(opt, channels, dataset)

    train_loader = get_loader_n3(image_root, gt_root, prior_dirs,
                                 batchsize=opt.training.batchsize,
                                 trainsize=opt.training.trainsize,
                                 num_workers=opt.training.num_workers,
                                 augmentation=online_aug)
                                 
    if args.debug:
        opt.training.epochs = 1

    for epoch in range(1, opt.training.epochs + 1):
        train(train_loader, model, optimizer, epoch, opt, debug=args.debug)
        scheduler.step()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

