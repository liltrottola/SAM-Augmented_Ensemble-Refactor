import random
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from SAM2UNet import SAM2UNet
from utils.dataloader import get_loader_with_aux   # aux: loader image + gt + aux instead of get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
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


# aux: Train_vanilla.test() is not copied: it is never called (commented out in train) and uses the loader without aux.
# The test of the aux models is done by Test_aux.py.

def train(train_loader, model, optimizer, epoch, opt, debug=False):
    model.train()
    # global best (mai usata)

    #original logic 
    size_rates = opt.training.size_rates
    clip_margin = opt.training.clip_margin
    #------------------------------------------------------------

    loss_record = AvgMeter()
    total_step = len(train_loader)
    
    for i, pack in enumerate(train_loader, start=1):
        #----DEBUG BLOCK ----
        if debug and i > 5:
            print("DEBUG MODE: breaking after 5 iterations")
            break
        #--------------------

        for rate in size_rates:
            # ---- data prepare ----
            images, gts, aux = pack   # aux: the loader also returns the aux image
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            aux = Variable(aux).cuda()

            # ---- rescale ----
            #uso dati da file yaml
            trainsize = int(round(opt.training.trainsize * rate / 32) * 32)

            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                aux = F.interpolate(aux, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # aux: two optimization steps per batch on the same model and the same gt:
            # first on the RGB image, then on the aux image

            # ---- forward (RGB) ----
            optimizer.zero_grad()
            pred0, pred1, pred2 = model(images)  # SAM2-UNet: final head + 2 side outputs (deep supervision)
            # ---- loss function ----
            loss0 = structure_loss(pred0, gts)
            loss1 = structure_loss(pred1, gts)
            loss2 = structure_loss(pred2, gts)
            loss = loss0 + loss1 + loss2
            # ---- backward ----
            loss.backward()
            if clip_margin is not None:
                # original SAM2-UNet has no gradient clipping (clip_margin: null in yaml)
                clip_gradient(optimizer, clip_margin)
            optimizer.step()

            # ---- forward (aux) ----
            optimizer.zero_grad()
            pred0, pred1, pred2 = model(aux)
            # ---- loss function ----
            loss0 = structure_loss(pred0, gts)
            loss1 = structure_loss(pred1, gts)
            loss2 = structure_loss(pred2, gts)
            loss = loss0 + loss1 + loss2
            # ---- backward ----
            loss.backward()
            if clip_margin is not None:
                clip_gradient(optimizer, clip_margin)
            optimizer.step()

            # ---- recording loss ---- (aux: loss of the aux step, as PolypPVT Train_aux)
            if rate == 1:
                loss_record.update(loss.data, opt.training.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}] lr'.format(
                      datetime.now(), epoch, opt.training.epochs, i, total_step,
                      loss_record.show()), optimizer.param_groups[0]['lr'])
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
    parser.add_argument("--config", type=str, default="../../../configs/sam2unet_aux.yaml", help="path to config file")
    parser.add_argument("--model_name", type=str, default=None, help="Model save name override")
    # aux: aux source and method, as PolypPVT Train_aux
    parser.add_argument('--aux_source', type=str, default=None, help='Folder under aux_root_base with the aux images (e.g. sam1, sam2, da3mono-large). If not specified, it will use experiment.aux_source defined in the configuration file.')
    parser.add_argument('--method', type=str, default=None, help='Method used for generating augmentation data. If not specified, it will use the default method defined in the configuration file.')
    parser.add_argument('--seed', type=int, default=None, help='Seed number for random number generation to ensure consistent results across runs.')
    parser.add_argument('--lr_method', type=str, default=None, help='Learning-rate strategy for ensemble diversity (lra/lrb/lrc). Overrides training.lr_method in the config. If not specified, uses the config value.')
    parser.add_argument('--offline_da', type=str, default=None, help='Offline DA method: da1/da2/clahe, or omit. Redirects images, masks and aux to the pre-generated augmented train set. Overrides training.offline_augmentation.')
    parser.add_argument('--online_da', type=str, default=None, help='Online DA method: da3, or omit. Applied at runtime by the aux dataloader. Overrides training.online_augmentation.')

    args = parser.parse_args()
    

    # 1. Load configuration
    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data) # Converte il dizionario in oggetto navigabile

    # aux: CLI overrides of aux source and method, used for the aux paths
    aux_source = args.aux_source if args.aux_source is not None else opt.experiment.aux_source
    method = args.method if args.method is not None else opt.experiment.method

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
    # SAM2-UNet: Hiera-L encoder initialized from the SAM2 checkpoint (frozen, only adapters + decoder are trained)
    model = SAM2UNet(opt.paths.hiera_path).cuda()
    params = model.parameters()

    # --- LR (LRa/LRb/LRc) for ensemble diversity ---
    # lr_cfg is None if training.lr_method is null/lrbase → legacy path (manual decay unchanged).
    lr_cfg = get_lr_method(getattr(opt.training, "lr_method", None))
    init_lr = lr_cfg["init_lr"] if lr_cfg is not None else opt.training.optimizer.lr

    if opt.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params, init_lr, weight_decay=opt.training.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, init_lr, weight_decay=1e-4, momentum=0.9)

    # lr_method lra/lrb/lrc: MultiStepLR as the other models.
    # legacy path (lrbase): original SAM2-UNet CosineAnnealingLR (T_max = epochs, eta_min), stepped once per epoch.
    if lr_cfg is not None:
        scheduler = build_scheduler(optimizer, lr_cfg)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.training.epochs, eta_min=opt.training.lr_schedule.eta_min)

    print(f"Optimizer configured: {opt.training.optimizer.type}, LR: {init_lr}, lr_method: {getattr(opt.training, 'lr_method', None)}")

    # --- DA: offline (da1/da2, redirects image_root) and online (da3, applied by the dataloader) are now independent axes ---
    offline_aug = getattr(opt.training, "offline_augmentation", None)
    online_aug = getattr(opt.training, "online_augmentation", None)
    dataset = opt.datasets.train[0]

    if offline_aug in OFFLINE_DA_METHODS:
        # aux: images and masks of the augmented train set, aux generated on those images
        # (<aux_source>/<method>/<da>/<dataset>); the test always uses the original images and aux
        image_root = '{}/{}/{}/images/'.format(opt.paths.offline_aug_base, offline_aug, dataset)
        gt_root    = '{}/{}/{}/masks/'.format(opt.paths.offline_aug_base, offline_aug, dataset)
        aux_root   = os.path.join(opt.paths.aux_root_base, aux_source, method, offline_aug, dataset, "images/")
        assert os.path.isdir(image_root), f"Offline DA folder not found: {image_root} -- run run_{offline_aug}_augmentation.py first"
        print(f"Using offline augmented data from: {image_root}")
    else:
        image_root = '{}/{}/images/'.format(opt.paths.datasets_root, dataset)
        gt_root    = '{}/{}/masks/'.format(opt.paths.datasets_root, dataset)
        # aux: assumes augmentation output uses the same folder name as the training dataset
        aux_root   = os.path.join(opt.paths.aux_root_base, aux_source, method, dataset, "images/")

    # aux: get_loader_with_aux instead of get_loader
    train_loader = get_loader_with_aux(image_root, gt_root, aux_root,
                                       batchsize=opt.training.batchsize,
                                       trainsize=opt.training.trainsize,
                                       num_workers=opt.training.num_workers,
                                       augmentation=online_aug)

    # aux: the loader pairs image and aux by position in the sorted lists,
    # so check that the names match one to one (as PolypPVT Train_aux)
    print(f"Aux images: {aux_root} ({len(train_loader.dataset.aux)} files)")
    image_names = [os.path.splitext(os.path.basename(f))[0] for f in train_loader.dataset.images]
    aux_names = [os.path.splitext(os.path.basename(f))[0] for f in train_loader.dataset.aux]
    if image_names != aux_names:
        mismatch = [(i, a) for i, a in zip(image_names, aux_names) if i != a]
        print(f"ERROR: IMAGE/AUX NAMES DO NOT MATCH ({len(image_names)} images, {len(aux_names)} aux, "
              f"{len(mismatch)} mismatched pairs), first: {mismatch[:5]}")
        exit(1)

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

        train(train_loader, model, optimizer, epoch, opt, debug=args.debug)

        # one scheduler step per epoch: cosine (legacy, as original) or MultiStepLR (lra/lrb/lrc)
        scheduler.step()
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

if __name__ == '__main__':

    # Set working directory to the script's location for consistent relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
   
