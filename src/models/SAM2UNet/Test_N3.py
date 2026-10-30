import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
#from scipy import misc
import yaml
from SAM2UNet_n3 import SAM2UNet_N3                # N3: wrapper ChannelReduction(N) -> SAM2UNet() instead of SAM2UNet
from utils.dataloader_n3 import test_dataset_n3    # N3: test loader RGB + K priors instead of test_dataset

from PIL import Image
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # to import from src
from src.sweep import registry as reg   # N3: prior families (folder + priors) of configs/sweep_registry.yaml

# -------- configuration from yaml file -----------
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


def build_prior_dirs(opt, channels, dataset):
    """N3: as Train_N3.build_prior_dirs without DA (the test always uses the original priors):
    one folder per channel "<folder>/<prior>", in the channel order: <prior_root>/<folder>/<dataset>/<prior>/"""
    prior_dirs = []
    for channel in channels:
        folder, prior = channel.split("/")
        prior_dirs.append(os.path.join(opt.paths.prior_root, folder, dataset, prior))
    return prior_dirs


def main():
    #1 parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../configs/sam2unet_n3.yaml', help='path to config file')

    #2 override for launcher
    parser.add_argument('--model_pth', type=str, default=None, help='Override model pth')
    parser.add_argument('--test_dataset', type=str, default=None, help='Override test dataset path')
    parser.add_argument('--save_path', type=str, default=None, help='Override save path for predictions')
    # N3: same prior channels as in training, as Train_N3 (if at least one is given they replace n3.prior_sources of the YAML)
    parser.add_argument('--sam_priors', type=str, nargs='+', default=None, help='N3: SAM priors used as channels. With the other two flags, replaces n3.prior_sources.')
    parser.add_argument('--depth_priors', type=str, nargs='+', default=None, help='N3: depth priors used as channels. With the other two flags, replaces n3.prior_sources.')
    parser.add_argument('--dsine_priors', type=str, nargs='+', default=None, help='N3: DSINE priors used as channels. With the other two flags, replaces n3.prior_sources.')
    parser.add_argument('--registry', type=str, default='../../../configs/sweep_registry.yaml', help='N3: registry with the prior families (folder + priors).')
    # N3: as in Train_N3, the same block type used in training (not in the weights: normal and residual have the same
    # parameters, only the forward changes). The kernel is read from the checkpoint.
    parser.add_argument('--residual', action='store_true', help='N3: residual block. Overrides n3.residual (sets it to true).')

    #3 parse args
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: CONFIGURATION FILE NOT FOUND: {args.config}")
        exit(1)

    cfg_data = load_config(args.config)
    opt = Config(cfg_data)

    # N3: prior channels as "<folder>/<prior>", from the flags if given, otherwise from n3.prior_sources of the YAML,
    # with the same function (checks and order) as Train_N3
    registry = reg.load_registry(args.registry)
    cli_priors = {"sam": args.sam_priors, "depth": args.depth_priors, "dsine": args.dsine_priors}
    channels = reg.get_n3_priors(registry, cfg_data['n3'], cli_priors)
    residual = True if args.residual else getattr(opt.n3, "residual", False)
    in_channels = 3 + len(channels)

    # Override with command line arguments if provided
    if args.model_pth is not None:
        model_pth = os.path.join(opt.paths.models_dir, args.model_pth)
    else:
        model_pth = os.path.join(opt.paths.models_dir, opt.testing.test_checkpoint)

    test_datasets = [args.test_dataset] if args.test_dataset is not None else opt.datasets.test
    save_path = args.save_path if args.save_path is not None else opt.paths.prediction_dir
    testsize = opt.testing.testsize

    model_name = os.path.splitext(os.path.basename(model_pth))[0]
    opt.model_name = model_name

    #4 load model
    # N3: SAM2UNet_N3(N, residual, k) instead of SAM2UNet(); from the first conv of the block of the checkpoint,
    # (32, N, k, k): N must match the test channels, k is taken as it is (the block is rebuilt with the kernel used in training)
    state_dict = torch.load(model_pth)
    first_conv = state_dict["reduce.channel_reduction.0.weight"]   # (32, N, k, k)
    ckpt_in_channels, kernel_size = first_conv.shape[1], first_conv.shape[-1]
    if ckpt_in_channels != in_channels:
        raise ValueError(f"{model_pth} was trained with N = {ckpt_in_channels} channels, "
                         f"but the test channels are RGB + {channels} (N = {in_channels})")
    print(f"N3 model {model_name}: channels RGB + {channels} -> N = {in_channels}, residual = {residual}, "
          f"kernel_size = {kernel_size} (from the checkpoint)")
    # no hiera checkpoint at test: all weights (frozen encoder included) come from the .pth, as original test.py
    model = SAM2UNet_N3(in_channels, residual, kernel_size)
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    scores = []
    #5 test loop
    for _data_name in test_datasets:
        data_path = os.path.join(opt.paths.datasets_root, _data_name)
        save_dir = os.path.join(save_path, model_name, _data_name)
        os.makedirs(save_dir, exist_ok=True)

        logits_dir = os.path.join(opt.paths.logits_dir, model_name, _data_name)
        os.makedirs(logits_dir, exist_ok=True)

        image_root = os.path.join(data_path, "images/")
        gt_root = os.path.join(data_path, "masks/")
        # N3: priors of this test dataset; same channels, same order and same treatment (prior_stats=None) as in training
        prior_dirs = build_prior_dirs(opt, channels, _data_name)
        test_loader = test_dataset_n3(image_root, gt_root, prior_dirs, testsize)
        # N3: number of images paired by name (the loader stops if a GT or a prior is missing), not the number of masks files
        num1 = test_loader.size

        DSC = 0.0 #to accumulate dice scores

        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            with torch.no_grad():   # as original SAM2-UNet test.py
                res, _, _ = model(image)  # only the first output (final head) is used at test
            #original code: res = F.upsample(..) now deprecated, replaced with interpolate
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)

            #we save logits and then apply sigmoid before saving
            logits_np = res.data.cpu().numpy().squeeze()
            np.save(os.path.join(logits_dir, name.replace('.png', '.npy')), logits_np)

            #apply sigmoid to get probabilities before saving as image
            res = res.sigmoid().data.cpu().numpy().squeeze()

            to_pil = transforms.ToPILImage()
            pil_img = to_pil(res)
            pil_img.save(os.path.join(save_dir, name))

            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(res, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (res.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

        score = DSC / num1
        scores.append(score)

        print(_data_name, 'Finish!')
        print(_data_name, score)

    print('Average DSC across datasets:', sum(scores)/len(scores))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
