import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader_with_aux, test_dataset_with_aux
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

seednumber = 37
random.seed(seednumber)
np.random.seed(seednumber)
torch.manual_seed(seednumber)
torch.cuda.manual_seed_all(seednumber)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def structure_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def test(model, dataset):
    data_path = os.path.join(opt.test_path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    aux_path = os.path.join(opt.aux_path, 'TestDataset', dataset)
    aux_root = '{}/images/'.format(aux_path)

    model.eval()
    num_test_samples = len(os.listdir(gt_root))
    testsize = 352
    test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, testsize)
    DSC = 0.0
    for i in range(num_test_samples):
        image, gt, aux, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        aux = aux.cuda()

        res, res1  = model(image)
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        aux_res, aux_res1  = model(aux)
        aux_res = F.upsample(aux_res + aux_res1 , size=gt.shape, mode='bilinear', align_corners=False)
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
        target = np.array(gt)

        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        smooth = 1
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num_test_samples



def train(train_loader, model, optimizer, epoch):
    model.train()
    size_rates = [0.75, 1, 1.25] 
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            # ---- data prepare ---- TODO: TRY TO REMOVE Variable
            images, gts, aux = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            aux = Variable(aux).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                aux = F.upsample(aux, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
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
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            optimizer.zero_grad()
            # ---- forward ----
            P1, P2= model(aux)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            # TODO: TRY TO REMOVE clip_gradient
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model
   
    mean_dice = 0
    for dataset in name:
        dataset_dice = test(model, dataset)
        mean_dice += dataset_dice
        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
        print(dataset, ': ', dataset_dice)
    print('Mean Performance: ', mean_dice/len(name))
    torch.save(model.state_dict(), save_path + opt.name_save)
    
if __name__ == '__main__':
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    ##################model_name#############################
    model_name = 'PolypPVT_with_SAMAug_seed_37'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation', type=int,
                        default=False, help='choose to do random flip rotation')

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
                        default='../../all_datasets_SAMAug2/TrainDataset',
                        help='path to train dataset')
    
    parser.add_argument('--aux_path', type=str,
                        default='/home/carisilore/Documents/RT/all_datasets_SAMAug2/',
                        help='path to SAM augmented images dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='../../all_datasets_SAMAug2/TestDataset',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/')
    
    parser.add_argument('--name_save', type=str,
                        default='HSNet.pth')
    
    opt = parser.parse_args()

    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PolypPVT().cuda()

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    aux_root = '{}/TrainDataset/images/'.format(opt.aux_path)

    train_loader = get_loader_with_aux(image_root, gt_root, aux_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch)
