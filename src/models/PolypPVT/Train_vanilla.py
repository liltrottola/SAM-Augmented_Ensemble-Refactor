import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import random
import logging
import yaml

#import matplotlib.pyplot as plt


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
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, opt, debug=False):
    model.train()

    size_rates = opt.training.size_rates
    loss_P2_record = AvgMeter()
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        #----DEBUG BLOCK ----
        if debug and i > 5:
            print("DEBUG MODE: breaking after 5 iterations")
            break
        #--------------------

        for rate in size_rates:
            # ---- data prepare ---- TODO: TRY TO REMOVE Variable
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
          
            # ---- rescale ----
            trainsize = int(round(opt.training.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            optimizer.zero_grad()
            # ---- forward ----
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()

            # TODO: TRY TO REMOVE clip_gradient
            clip_gradient(optimizer,  opt.training.clip_margin)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.training.batchsize)
        
        if i % 20 == 0 or i == total_step:
            #NOTE: qui era opt.epoch
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}] lr'.format(
                      datetime.now(), epoch, opt.training.epochs, i, total_step,
                      loss_P2_record.show()), optimizer.param_groups[0]['lr'])
    
     # save model 
    save_path = (opt.paths.models_dir)
    os.makedirs(save_path, exist_ok=True)
    #torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    filename = f"{opt.model_name}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, filename))


def main():
     # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", action="store_true", help="attiva modalità per debug")
    parser.add_argument("--config", type=str, default="../../../configs/polypPVT_vanilla.yaml", help="path to config file")
    parser.add_argument("--model_name", type=str, default=None, help="Model save name override")
    parser.add_argument('--seed', type=int, default=None, help='Seed number for random number generation to ensure consistent results across runs.')
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
    model = PolypPVT().cuda()
    params = model.parameters()

    if opt.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.training.optimizer.lr, weight_decay=opt.training.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, opt.training.optimizer.lr, weight_decay=1e-4, momentum=0.9)

    print(f"Optimizer configurato: {opt.training.optimizer.type}, LR: {opt.training.optimizer.lr}")

    image_root = '{}/{}/images/'.format(opt.paths.datasets_root, opt.datasets.train[0])
    gt_root = '{}/{}/masks/'.format(opt.paths.datasets_root, opt.datasets.train[0])

    train_loader = get_loader(image_root, gt_root, 
                              batchsize=opt.training.batchsize, 
                              trainsize=opt.training.trainsize,
                              augmentation=opt.training.augmentation)
    total_step = len(train_loader)

    print(f"Dataset caricato. Batch per epoca: {total_step}")
    print("#" * 20, "Start Training", "#" * 20)


    # In modalità debug, facciamo finta che ci sia 1 sola epoca per non perdere tempo
    if args.debug:
        print("!!! ATTENZIONE: MODALITÀ DEBUG ATTIVA !!!")
        opt.training.epochs = 1

    #original paper
    #for epoch in range(1, opt.epoch):
    #     adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
    
    for epoch in range(1, opt.training.epochs + 1):
        # --- SCHEDULER MANUALE ORIGINALE ---
        if epoch in opt.training.lr_schedule.milestones:
            adjust_lr(optimizer, opt.training.lr_schedule.decay_factor)

        train(train_loader, model, optimizer, epoch, opt, debug=args.debug)
    
if __name__ == '__main__':
    # Set working directory to the script's location for consistent relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
   
