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
    torch.save(model.state_dict(), os.path.join(save_path, opt.model_name + '.pth'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/polypPVT_aux.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to run a shorter training loop')
    
    # Optional command-line arguments to override specific configuration values from the YAML file, for run_training.py, but they can be used also for Train.py if needed
    parser.add_argument('--sam_version' , type=int , default=None, help='Version of SAM to use for augmentation data (e.g., 1 or 2). If not specified, it will use the default version defined in the configuration file.')
    parser.add_argument('--method', type=str, default=None, help='Method used for generating augmentation data. If not specified, it will use the default method defined in the configuration file.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model to be trained. This helps in identifying the model configuration when saving and tracking experiments. If not specified, it will use the default model name defined in the configuration file.')
    parser.add_argument('--seed', type=int, default=None, help='Seed number for random number generation to ensure consistent results across runs. If not specified, it will use the default seed defined in the configuration file.')
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERRORE: FILE CONFIGURAZIONE NON TROVATO: {args.config}")
        exit(1)

    # Load the configuration from the specified YAML file
    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # CLI overrides, used for path generation and name of the model
    sam_version = args.sam_version if args.sam_version is not None else opt.experiment.sam_version
    method = args.method if args.method is not None else opt.experiment.method
    model_name = args.model_name if args.model_name is not None else opt.experiment.name
    opt.model_name = model_name
    
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
    aux_root   = os.path.join(opt.paths.aux_root_base, f"sam{sam_version}", method , opt.datasets.train[0], "images/")

    # Create a data loader to fetch training batches with the specified batch size and image dimensions.
    train_loader = get_loader_with_aux(image_root, gt_root, aux_root, 
                                       batchsize=opt.training.batchsize,
                                       trainsize=opt.training.trainsize,
                                       augmentation=opt.training.augmentation)

    # Determine the total number of steps in the training process.
    total_step = len(train_loader)
    print(f"Dataset loaded. Batches per epoch: {total_step}")

    # In modalità debug, facciamo finta che ci sia 1 sola epoca per non perdere tempo
    if args.debug:
        print("!!! ATTENZIONE: MODALITÀ DEBUG ATTIVA !!!")
        opt.training.epochs = 1
    
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