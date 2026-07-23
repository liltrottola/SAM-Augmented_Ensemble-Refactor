import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..' , '..' , '..')) # to import from src

from src.augmentation import da3

'''
DataLoader for polyp segmentation tasks, including handling SAM auxiliary data and optional augmentations. 
The PolypDataset_aux class is a custom dataset that loads images, ground truth masks, and augmented data, 
applies transformations, and ensures consistent augmentations across all data types.

The get_loader_with_aux function creates a DataLoader for the PolypDataset_aux, allowing for efficient batching and shuffling during training.
'''

class PolypDataset_aux(data.Dataset):
    """
    Custom Dataset for polyp segmentation tasks, including handling auxiliary data and optional augmentations.
    """

    def __init__(self, image_root, gt_root, aux_root, trainsize, augmentations):
        # Initialize the dataset by specifying the image, ground truth, and auxiliary data directories.
        # Set the size of the training images and whether to apply augmentations.
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)

        # Gather and sort all image file paths from the specified directory, filtering by common image extensions.
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]

        # Similarly, collect and sort the ground truth and auxiliary file paths, filtering by PNG extension.
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.aux = [aux_root + f for f in os.listdir(aux_root) if f.endswith('.png') or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.aux = sorted(self.aux)

        # Ensure that the lists of images, ground truth, and auxiliary data are synchronized.
        self.filter_files()

        # Determine the total number of images in the dataset.
        self.size = len(self.images)

        # If augmentations are enabled, set up a series of transformations to apply to the images and ground truth.
        # This includes resizing, random rotations, flips, color adjustments, and normalization.
        if self.augmentations=='da3':
            self.img_transform , self.gt_transform , self.aux_transform = da3.get_da3_transforms_aux(self.trainsize)
        else:
            # If no augmentations are specified, use basic transformations that resize, convert to tensor, and normalize.
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),                            # Only resizing, converting to tensor, and normalizing
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

            self.aux_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        # Load the image, ground truth, and auxiliary data for a given index.
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        aux = self.rgb_loader(self.aux[index])

        # Generate a random seed to ensure the same augmentation is applied consistently within a batch.
        seed = np.random.randint(4294967295)

        # Set the seed for both Python's random module and PyTorch to ensure reproducible transformations.
        random.seed(seed)
        torch.manual_seed(seed)

        # Apply image transformations if they are defined.
        if self.img_transform is not None:
            image = self.img_transform(image)

        # Repeat the seeding process for auxiliary data transformations to ensure consistency.
        random.seed(seed)
        torch.manual_seed(seed)
        if self.aux_transform is not None:
            aux = self.aux_transform(aux)

        # Repeat the seeding process for ground truth transformations.
        random.seed(seed)
        torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        # Return the transformed image, ground truth, and auxiliary data.
        return image, gt, aux


    def filter_files(self):
        assert len(self.images) == len(self.gts) == len(self.aux)
        images = []
        gts = []
        auxs = []
        for img_path, gt_path,aux_path in zip(self.images, self.gts, self.aux):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            aux = Image.open(aux_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                auxs.append(aux_path)
        self.images = images
        self.gts = gts
        self.aux = auxs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, aux, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),aux.resize((w, h), Image.NEAREST)
        else:
            return img, gt,aux

    def __len__(self):
        return self.size

def get_loader_with_aux(image_root, gt_root, aux_root, batchsize, trainsize, shuffle=True, num_workers=1, pin_memory=True, augmentation=None):

    # Create an instance of the dataset. This dataset will handle loading images, ground truth masks,
    # and auxiliary images from the specified directories. It will also apply necessary transformations
    # such as resizing and normalization.
    dataset = PolypDataset_aux(image_root, gt_root, aux_root, trainsize, augmentation)

    # Instantiate a DataLoader with the specified batch size and other parameters.
    # This DataLoader will handle batching the data, shuffling it if required, and using multiple worker threads
    # to load the data in parallel for faster training.
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,           # Whether to shuffle the data at each epoch
                                  num_workers=num_workers,   # Number of worker threads for data loading
                                  pin_memory=pin_memory)     # Whether to use pinned memory for faster data transfer to GPU

    # Return the configured DataLoader instance which can now be used in training or evaluation loops.
    return data_loader

class test_dataset_with_aux:
    def __init__(self, image_root, gt_root, aux_root, testsize):
        # Setting up the test size that will be used to resize the images later
        self.testsize = testsize

        # We're gathering all the image files from the given directory. We're only interested in files that end with .jpg or .png
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]

        # Similarly, we're collecting all the ground truth files, but this time we're looking for .tif or .png extensions
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

        # And for the auxiliary files, we're again looking for .tif or .png extensions in the specified directory
        self.aux = [aux_root + f for f in os.listdir(aux_root) if f.endswith('.tif') or f.endswith('.png')]

        # Now, let's make sure our lists of file paths are sorted in ascending order to maintain consistency
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.aux = sorted(self.aux)

        # Time to define how we'll transform the images before using them. We'll resize them to our test size,
        # convert them to tensors, and then normalize them with predefined mean and standard deviation values
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # Mean values for normalization
                                 [0.229, 0.224, 0.225])  # Standard deviation values for normalization
        ])

        # For the ground truth files, a simple conversion to tensor will suffice
        self.gt_transform = transforms.ToTensor()

        # The auxiliary files get a similar treatment as the images: resize, convert to tensor, but no normalization
        self.aux_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])

        # Let's keep track of how many images we have by setting the size attribute
        self.size = len(self.images)

        # Finally, we initialize an index at 0, which we'll use to iterate through the dataset later
        self.index = 0



    def load_data(self):
        # Start by loading the current image using our rgb_loader method, which reads the image from disk
        image = self.rgb_loader(self.images[self.index])

        # Transform the image by:
        # 1. Resizing it to the specified test size (default 352x352).
        # 2. Converting it to a tensor, which is a multi-dimensional array used for numerical operations in PyTorch.
        # 3. Normalizing the tensor using predefined mean and standard deviation values to standardize the pixel values.
        # After these transformations, we add an extra dimension to the tensor using unsqueeze(0).
        # The unsqueeze(0) function adds a new dimension at position 0, making the tensor shape [1, C, H, W].
        # This is necessary because PyTorch models expect input tensors to have a batch dimension, even if the batch size is 1.
        image = self.transform(image).unsqueeze(0)

        # Now, load the corresponding ground truth file using the binary_loader method
        gt = self.binary_loader(self.gts[self.index])

        # Similarly, load the auxiliary image using the rgb_loader method
        aux = self.rgb_loader(self.aux[self.index])

        # Transform the auxiliary image just like we did with the main image, and add an extra dimension
        aux = self.aux_transform(aux).unsqueeze(0)

        # Extract the file name of the current image by splitting the path
        name = self.images[self.index].split('/')[-1]

        # If the file name ends with .jpg, change the extension to .png
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        # Move to the next index for the next call to this method
        self.index += 1

        # Return the processed image, ground truth, auxiliary image, and the file name
        return image, gt, aux, name


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        self.filter_files()
        self.size = len(self.images)

        '''
        if self.augmentations == 'da1':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
        '''
        # If the specified augmentation method is 'da3', we apply a specific set of transformations defined in the da3 module.
        # Otherwise, we apply a default set of transformations that include resizing, converting to tensor,
        if self.augmentations == 'da3':
            self.img_transform , self.gt_transform = da3.get_da3_transforms(self.trainsize)  
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=1, pin_memory=True, augmentation=None):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
