import random
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt_n3 import HSNet_N3                    # N3: wrapper ChannelReduction(N) -> HSNet() instead of HSNet
from utils.dataloader_n3 import get_loader_n3      # N3: loader RGB + K priors instead of get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging

import matplotlib.pyplot as plt

'''
    To import the lr_schedules, we need to add the src folder to the path,
    since it is not in the same directory as Train.py
'''
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # to import from src
from src.training.lr_schedules import get_lr_method, build_scheduler
from src.augmentation import OFFLINE_DA_METHODS
from src.sweep import registry as reg   # N3: prior families (folder + priors) of configs/sweep_registry.yaml

import yaml


# -------- YAML configuration loading block -----------
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
# ------------------------------------------------------------


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


# N3: Train_vanilla.test() is not copied: it is never called (commented out in train) and uses the 3-channel loader.
# The test of the N3 models is done by Test_N3.py.


def build_prior_dirs(opt, channels, dataset, da=None):
    """N3: one folder per channel "<folder>/<prior>", in the channel order: <prior_root>/<folder>/[<da>/]<dataset>/<prior>/
    (da = offline DA: the priors generated on the DA train set, as the aux images of Train_aux)"""
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
    # global best (mai usata)

    #original logic 
    size_rates = opt.training.size_rates
    clip_margin = opt.training.clip_margin
    #------------------------------------------------------------

    loss_P2_record = AvgMeter()
    total_step = len(train_loader)
    
    for i, pack in enumerate(train_loader, start=1):
        #----DEBUG BLOCK ----
        if debug and i > 5:
            print("DEBUG MODE: breaking after 5 iterations")
            break
        #--------------------

        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            #uso dati da file yaml
            trainsize = int(round(opt.training.trainsize * rate / 32) * 32)

            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3, P4= model(images)

            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, clip_margin)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                # NOTE: was loss_P2_record.update(loss_P2.data, opt.batchsize)
                loss_P2_record.update(loss_P4.data, opt.training.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            # NOTE: was opt.epoch
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}] lr'.format(
                      datetime.now(), epoch, opt.training.epochs, i, total_step,
                      loss_P2_record.show()), optimizer.param_groups[-1]['lr'])  # N3: LR of HSNet (last group), the block has lr * reduce_lr_mult
    # save model
    save_path = opt.paths.models_dir
    os.makedirs(save_path, exist_ok=True)
    
    #mean_dice = 0
    #for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    #    dataset_dice = test(model, test_path, dataset)
    #    mean_dice += dataset_dice
    #    logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
    #    print(dataset, ': ', dataset_dice)
    #print('Mean Performance: ', mean_dice/len(['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']))
    filename = f"{opt.model_name}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, filename))


def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--config", type=str, default="../../../configs/hsnet_n3.yaml", help="path to config file")
    parser.add_argument("--model_name", type=str, default=None, help="Model save name override")
    parser.add_argument('--seed', type=int, default=None, help='Seed number for random number generation to ensure consistent results across runs.')
    parser.add_argument('--lr_method', type=str, default=None, help='Learning-rate strategy for ensemble diversity (lra/lrb/lrc). Overrides training.lr_method in the config. If not specified, uses the config value.')
    parser.add_argument('--offline_da', type=str, default=None, help='Offline DA method: da1/da2/clahe/light, or omit. Redirects images, masks and priors to the DA train set. Overrides training.offline_augmentation.')
    parser.add_argument('--online_da', type=str, default=None, help='N3: noda or da3 (geometric on all channels, photometric only on the RGB). Overrides training.online_augmentation.')
    parser.add_argument('--train_dataset', type=str, default=None, help='Train set: folder under datasets_root (offline DA: <offline_aug_base>/<da>/<train set>). Overrides datasets.train.')
    # N3: prior channels per family (families and their folder in prior_sources of the registry); if at least one is
    # given they replace n3.prior_sources of the YAML, e.g. --sam_priors segPrior boundaryPrior --depth_priors frst_prior
    parser.add_argument('--sam_priors', type=str, nargs='+', default=None, help='N3: SAM priors used as channels. With the other two flags, replaces n3.prior_sources.')
    parser.add_argument('--depth_priors', type=str, nargs='+', default=None, help='N3: depth priors used as channels. With the other two flags, replaces n3.prior_sources.')
    parser.add_argument('--dsine_priors', type=str, nargs='+', default=None, help='N3: DSINE priors used as channels. With the other two flags, replaces n3.prior_sources.')
    parser.add_argument('--registry', type=str, default='../../../configs/sweep_registry.yaml', help='N3: registry with the prior families (folder + priors).')
    # N3: residual block (output = RGB + block, last conv at zero); if absent, n3.residual of the YAML
    parser.add_argument('--residual', action='store_true', help='N3: residual block. Overrides n3.residual (sets it to true).')
    # N3: kernel of the first conv of the block; if absent, n3.kernel_size of the YAML
    parser.add_argument('--kernel_size', type=int, default=None, help='N3: kernel of the first conv of the block (1 or 3). Overrides n3.kernel_size.')
    # N3: LR multiplier of the block; if absent, n3.reduce_lr_mult of the YAML
    parser.add_argument('--reduce_lr_mult', type=float, default=None, help='N3: LR of the block = lr * reduce_lr_mult. Overrides n3.reduce_lr_mult.')

    args = parser.parse_args()
    

    # 1. Load configuration
    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data) # Converte il dizionario in oggetto navigabile

    model_name = args.model_name if args.model_name is not None else opt.experiment.name
    opt.model_name = model_name  # Update opt with the model name (used for saving and logging)
    if args.seed is not None:
        opt.experiment.seed = args.seed # Aggiorna opt con il seed se fornito da CLI

    # CLI override for the LR method; if absent, keep the YAML value
    if args.lr_method is not None:
        opt.training.lr_method = args.lr_method

    # CLI overrides for the DA axes (for DA × LR sweep); if absent, keep the YAML value
    if args.offline_da is not None:
        opt.training.offline_augmentation = args.offline_da
    if args.online_da is not None:
        opt.training.online_augmentation = args.online_da
    # train set (run_training.py --train_dataset); offline DA, aux images and priors follow it
    if args.train_dataset is not None:
        opt.datasets.train = [args.train_dataset]

    # N3: prior channels as "<folder>/<prior>", from the flags if given, otherwise from n3.prior_sources of the YAML;
    # same function (checks and order) as run_training.py / run_inference.py
    registry = reg.load_registry(args.registry)
    cli_priors = {"sam": args.sam_priors, "depth": args.depth_priors, "dsine": args.dsine_priors}
    channels = reg.get_n3_priors(registry, cfg_data['n3'], cli_priors)
    residual = True if args.residual else getattr(opt.n3, "residual", False)
    kernel_size = args.kernel_size if args.kernel_size is not None else getattr(opt.n3, "kernel_size", 1)
    reduce_lr_mult = args.reduce_lr_mult if args.reduce_lr_mult is not None else getattr(opt.n3, "reduce_lr_mult", 1)
    in_channels = 3 + len(channels)



    #setup logging
    logging.basicConfig(filename=opt.paths.log_file,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    #seed for reproducibility
    if hasattr(opt.experiment, 'seed'):
        seed = opt.experiment.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        print(f"Seed set to: {seed}")

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # N3: ChannelReduction(N) -> HSNet() instead of HSNet(); block and HSNet trained together (same optimizer and lr)
    model = HSNet_N3(in_channels, residual, kernel_size).cuda()
    print(f"N3 model: channels RGB + {channels} -> N = {in_channels}, residual = {residual}, "
          f"kernel_size = {kernel_size}, reduce_lr_mult = {reduce_lr_mult}")
    logging.info(f"N3 model {model_name}: channels RGB + {channels}, N = {in_channels}, residual = {residual}, "
                 f"kernel_size = {kernel_size}, reduce_lr_mult = {reduce_lr_mult}")

    # --- LR (LRa/LRb/LRc) for ensemble diversity ---
    # lr_cfg is None if training.lr_method is null/lrbase → legacy path (manual decay unchanged).
    lr_cfg = get_lr_method(getattr(opt.training, "lr_method", None))
    init_lr = lr_cfg["init_lr"] if lr_cfg is not None else opt.training.optimizer.lr

    # N3: two parameter groups: the block (random init) with lr * reduce_lr_mult, HSNet (pretrained) with lr.
    # adjust_lr and MultiStepLR multiply every group, so the ratio stays constant during training
    params = [
        {"params": model.reduce.parameters(), "lr": init_lr * reduce_lr_mult},
        {"params": model.hsnet.parameters()},   # lr = init_lr, passed to the optimizer
    ]
    print(f"LR: block = {init_lr * reduce_lr_mult}, HSNet = {init_lr}")

    if opt.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params, init_lr, weight_decay=opt.training.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, init_lr, weight_decay=1e-4, momentum=0.9)

    # MultiStepLR scheduler only for the new LR methods; legacy path keeps the manual decay in the loop.
    scheduler = build_scheduler(optimizer, lr_cfg) if lr_cfg is not None else None

    print(f"Optimizer configured: {opt.training.optimizer.type}, LR: {init_lr}, lr_method: {getattr(opt.training, 'lr_method', None)}")

    # --- DA: offline (da1/da2/clahe/light, redirects image_root and the priors) and online (da3, applied by the dataloader) ---
    offline_aug = getattr(opt.training, "offline_augmentation", None)
    online_aug = getattr(opt.training, "online_augmentation", None)
    dataset = opt.datasets.train[0]

    if offline_aug in OFFLINE_DA_METHODS:
        # images and masks of the DA train set, priors generated on those images
        # (<prior_root>/<folder>/<da>/<dataset>/<prior>); the test always uses the original images and priors
        image_root = '{}/{}/{}/images/'.format(opt.paths.offline_aug_base, offline_aug, dataset)
        gt_root    = '{}/{}/{}/masks/'.format(opt.paths.offline_aug_base, offline_aug, dataset)
        prior_dirs = build_prior_dirs(opt, channels, dataset, da=offline_aug)
        assert os.path.isdir(image_root), f"Offline DA folder not found: {image_root} -- run run_{offline_aug}_augmentation.py first"
        print(f"Using offline augmented data from: {image_root}")
    else:
        image_root = '{}/{}/images/'.format(opt.paths.datasets_root, dataset)
        gt_root    = '{}/{}/masks/'.format(opt.paths.datasets_root, dataset)
        prior_dirs = build_prior_dirs(opt, channels, dataset)
    for c, d in zip(channels, prior_dirs):
        assert os.path.isdir(d), f"Prior folder not found: {d} -- generate the priors of {c} on {os.path.dirname(d)} first"
        print(f"  prior {c}: {d}")

    # N3: get_loader_n3 instead of get_loader; priors not standardized for now (prior_stats=None);
    # online DA (da3): geometric transforms on all the channels, photometric only on the RGB
    train_loader = get_loader_n3(image_root, gt_root, prior_dirs,
                                 batchsize=opt.training.batchsize,
                                 trainsize=opt.training.trainsize,
                                 augmentation=online_aug)
    total_step = len(train_loader)

    print(f"Dataset loaded. Batches per epoch: {total_step}")
    print("#" * 20, "Start Training", "#" * 20)


    # In debug mode, pretend there is only 1 epoch to avoid wasting time
    if args.debug:
        print("!!! ATTENZIONE: MODALITÀ DEBUG ATTIVA !!!")
        opt.training.epochs = 1

    #for epoch in range(1, opt.epoch):
    #     adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
    for epoch in range(1, opt.training.epochs + 1):

        if lr_cfg is None:
            # --- ORIGINAL MANUAL SCHEDULER (legacy path, unchanged) ---
            if epoch in opt.training.lr_schedule.milestones:
                adjust_lr(optimizer, opt.training.lr_schedule.decay_factor)

        train(train_loader, model, optimizer, epoch, opt, debug=args.debug)

        if lr_cfg is not None:
            # new lr scheduler step (only for the new LR methods, if specified)
            scheduler.step()
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

if __name__ == '__main__':

    # Set working directory to the script's location for consistent relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
   
