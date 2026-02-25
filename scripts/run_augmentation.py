import os
import argparse
import yaml
import skimage
from skimage import transform
import numpy as np
from tqdm import tqdm
import sys

# Per importare i moduli da 'src' anche se siamo in 'scripts'
# Aggiunge la cartella superiore al path di Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation.sam_loader import load_sam_model
from src.augmentation import methods

def run_processing(config):
    print("Configurazione caricata:", config.keys())
    dataset_root = config['paths']['dataset_root']
    output_root = config['paths']['output_root']

    for sam_version in config['sam']['versions']:

        checkpoints_root = config['paths']['checkpoints_root']

        if sam_version == 1:
            path_to_sam_checkpoint=os.path.join(checkpoints_root,config['sam']['checkpoints']['v1'])
        elif sam_version == 2:
            path_to_sam_checkpoint=os.path.join(checkpoints_root,config['sam']['checkpoints']['v2'])

        mask_generator = load_sam_model(sam_version, path_to_sam_checkpoint, config['sam'])
        
        # .get() per sicurezza se la lista è vuota
        methods_list = config['augmentation'].get('methods') or []
        for chosen_aug in methods_list:

            #Ottieni la funzione di augmentazione dal modulo methods
            if hasattr(methods, chosen_aug): #cerco se il metodo esiste in methods.py
                aug_function = getattr(methods, chosen_aug)
            else:
                print(f"Attenzione: Il metodo '{chosen_aug}' non esiste in methods.py")
                continue

            #print("(!) chosen sam version: " , sam_version, "chosen method: ", aug_function.__name__)
            print(f"------ Running {aug_function.__name__} with SAM v{sam_version} ------")

            for cur_dir in config['datasets']['folders']:

                folder_path = os.path.join(dataset_root, cur_dir, "images")
                saving_path = os.path.join(output_root, "sam"+str(sam_version), aug_function.__name__)
                print(f"Processing dataset folder: {folder_path}")
                print(f"Saving augmented images to: {saving_path}")

                file_names = os.listdir(folder_path)

                count = 0
                for filename in file_names:
                    tI = skimage.io.imread(os.path.join(folder_path, filename))

                    shape=tI.shape
                    # Check if the input image is grayscale (i.e., has only two dimensions: height and width)
                    if len(shape) == 2:
                        # Convert the grayscale image to an RGB image by duplicating the single channel across three channels
                        tmp = np.stack([tI] * 3, axis=-1)
                        # Assign the converted image back to tI
                        tI = tmp
                        shape=tI.shape

                    #in case of RGB+alpha channel, we remove the alpha channel since is unused
                    if shape[2]==4:
                        print(filename,": wrong shape FIXED")
                        tmp=np.empty((shape[0], shape[1], 3), dtype=float)
                        tmp[:,:,0]=tI[:,:,0]
                        tmp[:,:,1]=tI[:,:,1]
                        tmp[:,:,2]=tI[:,:,2]
                        tI=tmp

                    if tI.dtype == np.float64:
                        tI = (tI * 255).astype(np.uint8)
                        print(filename,": wrong type FIXED")

                    std_dim = tuple(config['image_processing']['resize_dim'])

                    original_size=tI.shape
                    resized_flag=False
                    #resize the image if too large
                    if original_size[0]>=config['image_processing']['resize_threshold'] or  original_size[1]>=config['image_processing']['resize_threshold']:
                        resized_flag=True
                        tI = transform.resize(tI, std_dim, anti_aliasing=True)
                        tI=tI.astype(np.float32)
                        print(filename, ": resized to: ", tI.shape, "FIXED")

                    #replace here with the wanted augmentation method
                    aug_img = aug_function(tI, mask_generator)

                    if resized_flag:
                        aug_img = transform.resize(aug_img, original_size, anti_aliasing=True)

                    # Convert to uint8
                    aug_img_scaled = skimage.exposure.rescale_intensity(aug_img, in_range='image', out_range=(0, 1))
                    aug_img_uint8 = skimage.img_as_ubyte(aug_img_scaled)

                    # save result
                    full_saving_path= os.path.join(saving_path, cur_dir, "images")
                    os.makedirs(full_saving_path, exist_ok=True)
                    skimage.io.imsave(os.path.join(full_saving_path, filename), aug_img_uint8)

                    count = count + 1
                    print("(-) ",cur_dir, ": ", count,"/",len(file_names))


if __name__ == "__main__":

    # Find directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    #costruisci il path del file di configurazione
    config_path = os.path.abspath(os.path.join(current_dir, '../configs/augmentation.yaml'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help='Path to the config file.')
    opt = parser.parse_args()

    if not os.path.exists(opt.config):
        raise FileNotFoundError(f"Config file not found: {opt.config}")
    
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    run_processing(config)
