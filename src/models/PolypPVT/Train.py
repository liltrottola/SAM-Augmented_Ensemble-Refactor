import torch
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader_with_aux, test_dataset_with_aux
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import numpy as np
import random
import torch.nn.functional as F
# import torch.nn as nn
import logging
import yaml


'''
    import matplotlib.pyplot as plt --> only used in plot_train() which is dead code
    import torch.nn as nn           --> not used
    import pdb                      --> not used
    from torch.autograd import Variable --> deprecated
'''


# ------- YAML config ---------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
# ----------------------------------

def structure_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def l1_loss(pred, mask):
    return (pred - mask).abs().mean()

def test(model, opt, dataset):

    data_path = os.path.join(opt.paths.datasets_root, dataset)

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    aux_path = os.path.join(opt.paths.aux_root, dataset)
    aux_root = '{}/images/'.format(aux_path)

    print(image_root, gt_root, aux_root)

    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, opt.testing.testsize)
    DSC = 0.0
    for i in range(num1):
        image, gt, aux,name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        aux=aux.cuda()

        res, res1  = model(image)
        res = F.interpolate(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        augres, augres1  = model(aux)
        augres = F.interpolate(augres + augres1 , size=gt.shape, mode='bilinear', align_corners=False)
        augres = augres.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        indicator1=np.mean(np.abs(res-0.5))
        indicator2=np.mean(np.abs(augres-0.5))
        if indicator1>indicator2:
            input = res
        else:
            input = augres

        target = np.array(gt)
        # N = gt.shape
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
    # global best
    size_rates = opt.training.size_rates 
    loss_P2_record = AvgMeter()

    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        if debug and i > 5:
            print("DEBUG MODE: stopping after 5 batches")
            break    
        
        for rate in size_rates:
            # ---- data prepare ----
            images, gts , aux = pack
            images = (images).cuda()
            gts = (gts).cuda()
            aux = (aux).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.training.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                aux = F.interpolate(aux, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            optimizer.zero_grad()
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.training.clip_margin)
            optimizer.step()

            # ---- forward ----
            optimizer.zero_grad()
            P1, P2= model(aux)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.training.clip_margin)
            optimizer.step()



            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.training.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.training.epochs, i, total_step,
                         loss_P2_record.show()))
    # save model 
    save_path = (opt.paths.models_dir)
    os.makedirs(save_path, exist_ok=True)
    #torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    #mean_dice = 0
    #if (epoch + 1) % 1 == 0:#'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 
    #    #for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    #    for dataset in ['test']:
    #        dataset_dice = test(model, test_path, dataset)
    #        mean_dice += dataset_dice
    #        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
    #        print(dataset, ': ', dataset_dice)
    #print('Mean Performance: ', mean_dice/len(['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']))
    torch.save(model.state_dict(), os.path.join(save_path , opt.experiment.name + '.pth'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/polypvt_aux.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to run a shorter training loop')
    
    #Override config parameters with command-line arguments if provided 
    parser.add_argument('--aux_root', type=str, help='Path to SAM augmented images dataset (overrides config if provided)')
    parser.add_argument('--model_name', type=str, help='Model name override (overrides experiment.name in config)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (overrides config if provided)')
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    # Load the configuration from the specified YAML file
    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # Override config parameters with command-line arguments if provided
    if args.aux_root is not None:
        opt.paths.aux_root = args.aux_root

    if args.model_name is not None:
        opt.experiment.name = args.model_name

    if args.seed is not None:
        opt.experiment.seed = args.seed

    # Set a seed number for random number generation to ensure consistent results across runs
    seednumber = opt.experiment.seed

    # Set the seed for Python's built-in random number generator
    # This ensures that any operations using Python's random module will produce the same results each time
    random.seed(seednumber)  # python random generator

    # Set the seed for NumPy's random number generator
    # This ensures that any operations using NumPy's random functions will produce the same results each time
    np.random.seed(seednumber)  # numpy random generator

    # Set the seed for PyTorch's CPU-based random number generator
    # This ensures that operations involving random number generation on the CPU will be reproducible
    torch.manual_seed(seednumber)

    # Set the seed for PyTorch's CUDA-based random number generator
    # This ensures that operations involving random number generation on the GPU will be reproducible
    torch.cuda.manual_seed_all(seednumber)

    # Ensure deterministic behavior in cuDNN operations
    # This setting makes sure that cuDNN operations are deterministic, which means they will produce the same results
    # across runs, but may impact performance. This is necessary for reproducibility.
    torch.backends.cudnn.deterministic = True

    # Disable the cuDNN auto-tuner to maintain deterministic behavior
    # When set to False, cuDNN will not look for the optimal algorithm for your hardware,
    # which helps in ensuring that the results are the same every time, but might affect performance.
    torch.backends.cudnn.benchmark = False

    # Configure logging to record the training process details.
    # Log entries will be saved to a file named 'train_log_seed_10.log' with timestamps and messages.
    logging.basicConfig(filename=opt.paths.log_file,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    model = PolypPVT().cuda()

    # ---- build optimizer ----
    params = model.parameters() 

    if opt.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.training.optimizer.lr, weight_decay=opt.training.optimizer.weight_decay)
    else:
        #to modify, currently hardcoded to use weight_decay=1e-4 and momentum=0.9. Also lr is taken from opt.training.optimizer.lr but is of different type of optimizer

        optimizer = torch.optim.SGD(params, opt.training.optimizer.lr, weight_decay=1e-4, momentum=0.9)

    # Configure a learning rate scheduler to reduce the learning rate at specific epochs.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=opt.training.lr_schedule.milestones, 
                                                     gamma=opt.training.lr_schedule.gamma)

    # Set up paths for accessing training, ground truth, and auxiliary data.
    image_root = os.path.join(opt.paths.datasets_root, opt.datasets.train[0], "images/")
    gt_root    = os.path.join(opt.paths.datasets_root, opt.datasets.train[0], "masks/")

    # assumes augmentation output uses the same folder name as the training dataset
    aux_root   = os.path.join(opt.paths.aux_root, opt.datasets.train[0], "images/")

    # Create a data loader to fetch training batches with the specified batch size and image dimensions.
    train_loader = get_loader_with_aux(image_root, gt_root, aux_root, 
                                       batchsize=opt.training.batchsize,
                                       trainsize=opt.training.trainsize,
                                       augmentation=opt.training.augmentation)

    # Determine the total number of steps in the training process.
    total_step = len(train_loader)
    print(f"Dataset loaded. Batches per epoch: {total_step}")
    
    # Begin the training process, iterating through each epoch.
    for epoch in range(1, opt.training.epochs+1):
         # Call the training function to train the model for the current epoch.
         train(train_loader, model, optimizer, epoch, opt, debug=args.debug)
         
         # Update the learning rate using the scheduler.
         scheduler.step()
    

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

    '''
    #dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    #name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation', type=str,
                        default='da3', help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')
    
    parser.add_argument('--aux_path', type=str,
                        default='',
                        help='path to SAM augmented images dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')
    
    parser.add_argument('--name_save', type=str,
                        default='PolypPVT.pth')
    
    parser.add_argument('--loss', type=int, default=-1)

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    if opt.loss == -1:
        structure_loss = structure_loss
        print("Loss function: ", "structure_loss")
    else:
        structure_loss = newloss.losses[opt.loss]
        print("Loss function: ",newloss.losses[opt.loss].__name__)

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PolypPVT().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20],gamma=0.2)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    aux_root = '{}/train/images/'.format(opt.aux_path)
    
    #print(image_root, gt_root, aux_root)
    
    train_loader = get_loader_with_aux(image_root, gt_root, aux_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    #print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        #adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
        scheduler.step()
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
'''

''' 
structure_loss = structure_loss



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
    

'''