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