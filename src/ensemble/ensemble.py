import numpy as np
from torchvision import transforms
from PIL import Image
import os


# To evaluate the ensemble of networks, we first save the output predictions 
# from each individual network into separate folders. The main folder containing 
# these model output folders should be specified with the --models_output argument.
# In particular in the specified --models_output folder each subfolder represents 
# the output from a different model we intend to include in the ensemble.
#
# For each test image in the test set (provided by the --test_masks argument), 
# the following steps occur:
# 1. Retrieve the paths of corresponding output images from each model’s output folder. 
#    Here, n represents the number of models in the ensemble, so n images should be retrieved,
#    one from each model output folder, corresponding to the same test image.
# 2. Load each of the n images and sum them together pixel-wise to produce an aggregated 
#    output for that test image.


def process_images_in_folder(models_path, folder_to_save, dataset):
    models = os.listdir(models_path)
    files = os.listdir(os.path.join(models_path, os.path.join(models[0], dataset)))
    image_files = sorted(files)

    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    
    for i in range(len(image_files)):
        paths = []
        for m in models:
            paths.append(os.path.join(models_path, os.path.join(m, dataset, image_files[i])))
        
        mean = generate_mean(paths)
        file_n = os.path.join(folder_to_save, image_files[i])
        
        # Convert mean tensor to numpy array, squeeze, and change data type
        mean_image = (mean.squeeze().numpy() * 255).astype(np.uint8)
        Image.fromarray(mean_image).save(file_n)

#input: "paths" is the list containing the paths of every different image we want to get the mean of.
#This function use the average rule to combine the n images from the n networks, then it transforms the 
#obtained image in a binary mask and return it.
def generate_mean(paths):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    #open the images as grayscale
    images = [image_transform(Image.open(p).convert("L")) for p in paths]

    for i in range(1, len(images)):
        images[0] = images[0] + images[i]

    mean = images[0] / len(images)
    #apply threshold to get binary mask 
    #mean[mean < 0.5] = 0
    #mean[mean >= 0.5] = 1

    return mean

#calculate the dice score between the ensemble image and the gt 
def get_dice(labels_path, ensemble_prediction_path, dataset_name):
    #dice_metric = DiceMetric(include_background=True, reduction="mean")
    transform = transforms.ToTensor()
    
    files_gt = os.listdir(labels_path)
    outputs = []
    targets = []

    for gt in files_gt:
        n1 = os.path.join(labels_path, gt)
        n2 = os.path.join(ensemble_prediction_path, gt)
        
        i1 = Image.open(n1).convert("L")
        i2 = Image.open(n2).convert("L")

        i1 = transform(i1)
        i2 = transform(i2)
        #apply threshold to get binary image
        i1[i1>=0.5]=1
        i1[i1<0.5]=0        
        i2[i2>=0.5]=1
        i2[i2<0.5]=0

        outputs.append(i1)
        targets.append(i2)

    dices=[]
    for i in range(0, len(outputs)):
        #dices.append(dice_metric(outputs[i].unsqueeze(0), targets[i].unsqueeze(0)).item())
        input=outputs[i]
        target=targets[i]
        input=np.asarray(input, np.float32)
        input=input/(input.max()+1e-8)
        target=np.asarray(target, np.float32)
        target=target/(target.max()+1e-8)
        smooth = 1e-8
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        #dice = '{:.4f}'.format(dice)
        dice = float(dice)
        dices.append(dice)
        
    #for i, name in enumerate(files_gt):
    #    print(dataset_name, ",", name, ",", dices[i])

    print(f"{dataset_name} mDICE:", sum(dices) / len(dices))
    return sum(dices) / len(dices)