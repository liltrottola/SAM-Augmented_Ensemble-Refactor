import os
import argparse
import sys

# Per importare i moduli da 'src' anche se siamo in 'scripts'
# Aggiunge la cartella superiore al path di Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ensemble.ensemble import get_dice
from src.ensemble.ensemble import process_images_in_folder

def main():
    #get user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--models_outputs', type=str)
    parser.add_argument('--test_masks', type=str)
    opt = parser.parse_args()
    
    #each subfolder of --models_outputs contains the output for the whole 5 polyp datasets.  
    models_to_sum = os.listdir(opt.models_outputs)
    buffer = []

    model1_path = os.path.join(opt.models_outputs, models_to_sum[0])
    datasets = os.listdir(model1_path)
    for item in datasets:
        current_labels_path = os.path.join(opt.test_masks, item, "masks")
        current_output_folder = os.path.join(opt.out_folder, item, "mean")

        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        process_images_in_folder(opt.models_outputs, current_output_folder, item)         
        buffer.append(get_dice(current_labels_path, current_output_folder, item))
        
    #mean across all datasets
    print("mean", sum(buffer) / len(buffer))

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()