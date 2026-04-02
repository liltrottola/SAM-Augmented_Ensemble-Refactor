import torch
import os
import argparse
from datetime import datetime
from lib.pvt_aux import HSNet_with_aux
from utils.dataloader import get_loader_with_aux #, test_dataset_with_aux
from utils.utils import AvgMeter # clip_gradient and adjust_lr are not used in this code, so they are not imported
import numpy as np
import random
import torch.nn.functional as F
import logging
import yaml

'''
    from torch.autograd import Variable --> is deprecated
    import torch.nn as nn --> is not used in this code, so it is not imported
    import pdb ----> not used in this code, so it is not imported
    from torchvision import transforms ---> used only in test() and not in train() 
    from PIL import Image ---> used only in test()
    import matplotlib.pyplot as plt
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
    # Start by computing the binary cross-entropy loss between the predicted logits and the ground truth mask
    # The loss is computed per pixel without any reduction
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    
    # Apply the sigmoid function to the predicted logits to get probabilities in the range [0, 1]
    pred = torch.sigmoid(pred)
    
    # Compute the intersection between the predicted probabilities and the ground truth mask
    # This is done by element-wise multiplication followed by summing over the height and width dimensions
    inter = ((pred * mask)).sum(dim=(2, 3))
    
    # Compute the union of the predicted probabilities and the ground truth mask
    # This is done by element-wise addition followed by summing over the height and width dimensions
    union = ((pred + mask)).sum(dim=(2, 3))
    
    # Calculate the weighted Intersection over Union (IoU) metric
    # We subtract the intersection from the union to avoid counting overlapping areas twice
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    # Combine the weighted binary cross-entropy loss and the weighted IoU loss
    # The mean loss is computed across all pixels
    return (wbce + wiou).mean()

def l1_loss(pred, mask):
    return (pred - mask).abs().mean()

# This function is strongly interconnected with the main() and it uses many of its variables
def train(train_loader, model, optimizer, epoch, opt, debug=False):
    # Set the model to training mode, enabling behaviors like dropout and batch normalization
    model.train()
    
    # Define a list of size rates to experiment with different scales of the input images
    # These rates determine how much the input images will be rescaled during training
    size_rates = opt.training.size_rates
    
    # Create an AvgMeter instance to track and compute the average value of the loss metric for P2
    # This will help in monitoring the loss associated with the P2 output of the model
    loss_P4_record = AvgMeter()
    total_step = len(train_loader)
    
    # Loop through the training data loader, enumerating to get both index and data batch
    for i, pack in enumerate(train_loader, start=1):
        #debug mode: if the debug flag is set to True, we will stop the training loop after processing 5 batches
        if debug and i > 5:
            print("DEBUG MODE: stopping after 5 batches")
            break

        # Loop through each size rate to perform training with images of different scales
        for rate in size_rates:
            # ---- Prepare the data ----
            # Unpack the batch into images, ground truths, and auxiliary inputs
            images, gts, aux = pack
            # Move the data to GPU for faster processing
            images = (images).cuda()
            gts = (gts).cuda()
            aux = (aux).cuda()
            
            # ---- Rescale the images ----
            # Compute the new size for images based on the current size rate
            # This ensures that images are resized to a multiple of 32, which might be required by the model
            trainsize = int(round(opt.training.trainsize * rate / 32) * 32)
            
            # If the size rate is not 1 (original size), rescale the images and ground truths
            # This is done to create training data at different scales
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                aux = F.interpolate(aux, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- Forward pass ----
            # Zero out the gradients from the previous iteration
            optimizer.zero_grad()
            # Perform a forward pass through the model with the resized images
            P1, P2, P3, P4, P1_, P2_, P3_, P4_ = model(images)
            # Compute the loss for each output of the model using the structure_loss function
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            # Sum up the losses to get the total loss for this iteration
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # Perform backpropagation to compute the gradients
            loss.backward()
            # Update the model parameters using the optimizer
            optimizer.step()
            
            # Zero out the gradients again before the next forward pass
            optimizer.zero_grad()
            # Perform a forward pass through the model with the auxiliary inputs
            P1, P2, P3, P4, P1_, P2_, P3_, P4_ = model(aux)
            # Compute the loss for each output using the auxiliary inputs
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            # Sum up the losses to get the total loss for this iteration
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # Perform backpropagation to compute the gradients
            loss.backward()
            # Update the model parameters using the optimizer
            optimizer.step()
            
            # ---- Record the loss ----
            # If the current size rate is 1 (original size), update the AvgMeter with the loss value
            if rate == 1:
                # Update the loss_P2_record with the current loss for P4
                loss_P4_record.update(loss_P4.data, opt.training.batchsize)
        
        # ---- Training visualization ----
        # Every 20 steps or at the end of the epoch, print the current training status
        # This includes the average loss for P4 (last 40 iterations) and the learning rate
        # $epoch is a variable from the '__main__', $i is the number of batch processed within the epoch
        # The loss shown is the $structure_loss
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  'lateral-5: {:0.4f}] lr'.
                  format(datetime.now(), epoch, opt.training.epochs, i, total_step,
                         loss_P4_record.show()), optimizer.param_groups[0]['lr'])
    
    # Creating the specified directory to save the model state
    save_path = (opt.paths.models_dir)
    os.makedirs(save_path, exist_ok=True)
    
    """
    mean_dice = 0
    # Test the model on multiple datasets after each epoch
    for dataset in name:
        # Evaluate the model on each dataset and log the Dice score
        dataset_dice = test(model, dataset)
        mean_dice += dataset_dice
        # Log the current epoch, dataset name, and the computed Dice score for the dataset
        # This helps in tracking the model's performance across different datasets and epochs
        # The log entry is saved to a file or console output for later review and analysis
        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
        print(dataset, ': ', dataset_dice)
        dict_plot[dataset].append(dataset_dice)
    print('Mean Performance: ', mean_dice/len(name))
    """
    #if epoch == opt.epoch - 1
        # Last epoch reached, save the resulting model
    torch.save(model.state_dict(), os.path.join(save_path, opt.model_name + '.pth'))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/hsnet_aux.yaml', help='Path to the YAML configuration file')
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
    
    # Instantiate the model and move it to the GPU for training.
    model = HSNet_with_aux().cuda()

    # Set up the optimizer to adjust the model's parameters during training.
    # Using Adam optimizer with the learning rate specified in the arguments.
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.training.optimizer.lr, weight_decay=opt.training.optimizer.weight_decay)

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
        was in __main__ but it is strongly interconnected with the train() function. to be evaluated if it is better to keep it 
        # Initialize a dictionary to store Dice scores for various datasets.
        # Each dataset name is mapped to an empty list which will later hold the Dice scores.
        dict_plot = {'test':[]}
        # Define a list of dataset names for easy reference.
        name = ['test']

        # Set the name of the model to be used. This helps in identifying the model configuration 
        # when saving and tracking experiments.
        model_name = 'hsnet_with_aux'
    '''

'''
def test(model, dataset):
    # We start by constructing the path to the dataset based on the provided path and dataset name
    data_path = os.path.join(opt.test_path, dataset)

    # Define where the images and ground truth masks are located within the dataset
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    
    aux_path = os.path.join(opt.aux_path, 'TestDataset', dataset)

    # In test mode, we use a directory specific to the current dataset
    aux_root = '{}/images/'.format(aux_path)

    # Set the model to evaluation mode, which turns off certain layers like dropout that are only needed during training
    model.eval()
    
    # Count the number of ground truth masks to determine how many samples we have to test
    num_test_samples = len(os.listdir(gt_root))
    
    # We'll resize all images to a consistent size of 352x352 pixels
    testsize = 352
    
    # Initialize the test data loader with the specified image, mask, and auxiliary image paths, as well as the test size
    test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, testsize)
    
    # Initialize a variable to accumulate the Dice Similarity Coefficient (DSC) across all samples
    DSC = 0.0
    
    # Loop through each sample in the dataset
    for i in range(num_test_samples):
        # Load the current image, ground truth mask, auxiliary image, and the image name using the test_loader
        image, gt, aux, name = test_loader.load_data()
        
        # Convert the ground truth mask to a numpy array of type float32 for further processing
        gt = np.asarray(gt, np.float32)
        
        # Normalize the ground truth mask to ensure its values are between 0 and 1
        gt /= (gt.max() + 1e-8)
        
        # Move the image and auxiliary image to the GPU to speed up computation
        image = image.cuda()
        aux = aux.cuda()
        
        # Perform a forward pass with the main image through the model, obtaining multiple levels of output
        res, res1, res2, res3, _, _, _, _ = model(image)
        
        # Combine the outputs and interpolate the result to match the shape of the ground truth mask
        res = F.interpolate(res + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        
        # Apply the sigmoid function to convert the model outputs to probabilities, and move the result back to the CPU
        res = res.sigmoid().data.cpu().numpy().squeeze()
        
        # Normalize the result to ensure the values are between 0 and 1
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # Do the same forward pass and processing for the auxiliary image
        aux_res, aux_res1, aux_res2, aux_res3, _, _, _, _ = model(aux)
        aux_res = F.interpolate(aux_res + aux_res1 + aux_res2 + aux_res3, size=gt.shape, mode='bilinear', align_corners=False)
        aux_res = aux_res.sigmoid().data.cpu().numpy().squeeze()
        #aux_res = (aux_res - aux_res.min()) / (aux_res.max() - aux_res.min() + 1e-8)
        
        # Calculate an indicator for both the main and auxiliary results, measuring how far the predictions are from 0.5 on average
        indicator1 = np.mean(np.abs(res - 0.5))
        indicator2 = np.mean(np.abs(aux_res - 0.5))
        
        # Choose the prediction that is more confident (less centered around 0.5)
        if indicator1 > indicator2:
            input = res
        else:
            input = aux_res
        
        # Flatten the ground truth mask and the chosen prediction for easier calculation of the Dice coefficient
        target = np.array(gt)
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        
        # Calculate the Dice coefficient, which measures the overlap between the prediction and the ground truth
        intersection = (input_flat * target_flat)
        smooth = 1  # Add a small value to avoid division by zero
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        
        # Format the Dice coefficient to four decimal places and convert it to a float
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        
        # Accumulate the Dice coefficient to compute the average later
        DSC = DSC + dice
    
    # Return the average Dice coefficient across all samples
    return DSC / num_test_samples 
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


'''