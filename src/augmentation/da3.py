import torchvision.transforms as transforms

def get_da3_transforms(trainsize):

    img_transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.RandomRotation(90),                   # Apply random rotation up to 90 degrees
        transforms.RandomVerticalFlip(p=0.5),            # Apply random vertical flip with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),          # Apply random horizontal flip with 50% probability
        transforms.RandomGrayscale(p=0.5),               # Randomly convert to grayscale with 50% probability
        transforms.RandomInvert(p=0.5),                   # Randomly invert colors with 50% probability
        transforms.RandomAutocontrast(p=0.2),             # Randomly adjust contrast with a 20% probability
        transforms.RandomEqualize(p=0.2),                 # Randomly equalize image histogram with 20% probability
        transforms.ToTensor(),                            # Convert image to tensor format
        transforms.Normalize([0.485, 0.456, 0.406],     # Normalize tensor with pre-defined mean and std dev
                            [0.229, 0.224, 0.225])])

    gt_transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.RandomRotation(90),                   # Same augmentation applied to ground truth
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
    
    return img_transform , gt_transform

def get_da3_geometric_transforms():
    '''
    N3: only the geometric part of get_da3_transforms, for the prior channels (photometric changes only on the RGB).
    Applied on the (K, trainsize, trainsize) prior tensor, already resized: the 3 random draws (angle, vertical flip,
    horizontal flip) come in the same order as in img_transform / gt_transform, whose geometric transforms come before
    the photometric ones, so with the same seed RGB, GT and priors get the same rotation and flips.
    '''
    prior_transform = transforms.Compose([
        transforms.RandomRotation(90),                   # same angle as image and gt (nearest, fill 0)
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5)])

    return prior_transform

def get_da3_transforms_aux(trainsize):
    print('Using DA3:  RandomRotation, RandomFlip, color changes')
    img_transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.RandomRotation(90),                   # Apply random rotation up to 90 degrees
        transforms.RandomVerticalFlip(p=0.5),            # Apply random vertical flip with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),          # Apply random horizontal flip with 50% probability
        transforms.RandomGrayscale(p=0.5),               # Randomly convert to grayscale with 50% probability
        transforms.RandomInvert(p=0.5),                   # Randomly invert colors with 50% probability
        transforms.RandomAutocontrast(p=0.2),             # Randomly adjust contrast with a 20% probability
        transforms.RandomEqualize(p=0.2),                 # Randomly equalize image histogram with 20% probability
        transforms.ToTensor(),                            # Convert image to tensor format
        transforms.Normalize([0.485, 0.456, 0.406],     # Normalize tensor with pre-defined mean and std dev
                            [0.229, 0.224, 0.225])])

    gt_transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.RandomRotation(90),                   # Same augmentation applied to ground truth
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])

    aux_transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.RandomRotation(90),                   # Same augmentation applied to auxiliary data
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomInvert(p=0.5),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    
    return img_transform , gt_transform , aux_transform