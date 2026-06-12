import random
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt_vanilla import HSNet
from utils.dataloader import get_loader, test_dataset
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

import yaml

# -------- blocco per configurazioni da file yaml -----------
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


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2, res3  = model(image)
        # eval Dice
        res = F.interpolate(res + res1 + res2 + res3 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

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
                #NOTE: qui era loss_P2_record.update(loss_P2.data, opt.batchsize)
                loss_P2_record.update(loss_P4.data, opt.training.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            #NOTE: qui era opt.epoch
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}] lr'.format(
                      datetime.now(), epoch, opt.training.epochs, i, total_step,
                      loss_P2_record.show()), optimizer.param_groups[0]['lr'])
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
    # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", action="store_true", help="attiva modalità per debug")
    parser.add_argument("--config", type=str, default="../../../configs/hsnet_vanilla.yaml", help="path to config file")
    parser.add_argument("--model_name", type=str, default=None, help="Model save name override")
    parser.add_argument('--seed', type=int, default=None, help='Seed number for random number generation to ensure consistent results across runs.')
    parser.add_argument('--lr_method', type=str, default=None, help='Learning-rate strategy for ensemble diversity (lra/lrb/lrc). Overrides training.lr_method in the config. If not specified, uses the config value.')
    parser.add_argument('--augmentation', type=str, default=None, help='DA method: da1/da2 (offline), da3 (online), or omit. Overrides training.augmentation.')

    args = parser.parse_args()
    

    # 1. Carica Configurazione
    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data) # Converte il dizionario in oggetto navigabile

    model_name = args.model_name if args.model_name is not None else opt.experiment.name
    opt.model_name = model_name # Aggiorna opt con il nome del modello (utile per salvataggio e logging)
    if args.seed is not None:
        opt.experiment.seed = args.seed # Aggiorna opt con il seed se fornito da CLI

    # CLI override del metodo LR ; se assente resta il valore del YAML
    if args.lr_method is not None:
        opt.training.lr_method = args.lr_method

    # CLI override del metodo DA (per lo sweep DA × LR); se assente resta il valore del YAML
    if args.augmentation is not None:
        opt.training.augmentation = args.augmentation


    #setup logging
    logging.basicConfig(filename=opt.paths.log_file,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    #seed per riproducibilità (non presente in HSNet originale)
    if hasattr(opt.experiment, 'seed'):
        seed = opt.experiment.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        print(f"Seed impostato a: {seed}")

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = HSNet().cuda()
    params = model.parameters()

    # --- LR (LRa/LRb/LRc) per diversità d'ensemble ---
    # lr_cfg è None se training.lr_method è null → path legacy (decay manuale invariato).
    lr_cfg = get_lr_method(getattr(opt.training, "lr_method", None))
    init_lr = lr_cfg["init_lr"] if lr_cfg is not None else opt.training.optimizer.lr

    if opt.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params, init_lr, weight_decay=opt.training.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, init_lr, weight_decay=1e-4, momentum=0.9)

    # Scheduler MultiStepLR solo per i metodi LR nuovi; nel legacy resta il decay manuale nel loop.
    scheduler = build_scheduler(optimizer, lr_cfg) if lr_cfg is not None else None

    print(f"Optimizer configurato: {opt.training.optimizer.type}, LR: {init_lr}, lr_method: {getattr(opt.training, 'lr_method', None)}")

    # --- DA (DA1/DA2/DA3) ---
    # da1/da2 = offline (immagini gia' aumentate su disco), da3 = online, null = nessuna aug.
    aug = opt.training.augmentation
    dataset = opt.datasets.train[0]

    # If DA offline, set image_root and gt_root to the augmented data folders; if online or no DA, use original dataset folders.
    if aug in OFFLINE_DA_METHODS:
        image_root = '{}/{}/{}/{}/images/'.format(opt.paths.offline_aug_base, aug, opt.training.da_source, dataset)
        gt_root    = '{}/{}/{}/{}/masks/'.format(opt.paths.offline_aug_base, aug, opt.training.da_source, dataset)
        assert os.path.isdir(image_root), f"Offline DA folder not found: {image_root} -- run run_{aug}_augmentation.py first"
        online_aug = None    # images already augmented offline, no online augmentation
        print(f"Using offline augmented data from: {image_root}")
        
    else:                    # 'da3' (online) o None
        image_root = '{}/{}/images/'.format(opt.paths.datasets_root, dataset)
        gt_root    = '{}/{}/masks/'.format(opt.paths.datasets_root, dataset)
        online_aug = aug

    train_loader = get_loader(image_root, gt_root,
                              batchsize=opt.training.batchsize,
                              trainsize=opt.training.trainsize,
                              augmentation=online_aug)
    total_step = len(train_loader)

    print(f"Dataset caricato. Batch per epoca: {total_step}")
    print("#" * 20, "Start Training", "#" * 20)


    # In modalità debug, facciamo finta che ci sia 1 sola epoca per non perdere tempo
    if args.debug:
        print("!!! ATTENZIONE: MODALITÀ DEBUG ATTIVA !!!")
        opt.training.epochs = 1

    #for epoch in range(1, opt.epoch):
    #     adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
    for epoch in range(1, opt.training.epochs + 1):

        if lr_cfg is None:
            # --- SCHEDULER MANUALE ORIGINALE (path legacy, invariato) ---
            if epoch in opt.training.lr_schedule.milestones:
                adjust_lr(optimizer, opt.training.lr_schedule.decay_factor)

        train(train_loader, model, optimizer, epoch, opt, debug=args.debug)

        if lr_cfg is not None:
            # new lr scheduler step (solo per i metodi LR nuovi, se specificati)
            scheduler.step()
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

if __name__ == '__main__':

    # Set working directory to the script's location for consistent relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
   
