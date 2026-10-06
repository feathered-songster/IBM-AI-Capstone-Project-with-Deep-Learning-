from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def create_dataloaders(batch_size):
    # --------------------------------------------------------
    # Data locations as prepared in prepare_data.py
    # --------------------------------------------------------

    train_dir = Path("./data/train")
    val_dir = Path("./data/val")
    test_dir = Path("./data/test")


    # --------------------------------------------------------
    # Training transformation contains data augmentation,
    # Validation transformation does not, use for test data too
    # --------------------------------------------------------

    train_transform = transforms.Compose([ #transforms.Resize((64, 64)), # not really needed here as pictures are already 64x64
                                         transforms.RandomHorizontalFlip(p=0.5), # probability 0.5
                                         transforms.RandomVerticalFlip(p=0.5),   # probability 0.5
                                         transforms.RandomRotation(45),
                                         transforms.RandomAffine(0, shear=0.2),
                                         transforms.ToTensor(), # Converts PIL Image to a tensor, reorders to (channels, height, width) and scales values to [0, 1]
                                         transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # Normalize from [0, 1] to [-1, 1]
    ])

    val_transform = transforms.Compose([ #transforms.Resize((64, 64)), # not really needed here as pictures are already 64x64
                                         transforms.ToTensor(), # Converts PIL Image to a tensor, reorders to (channels, height, width) and scales values to [0, 1]
                                         transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # Normalize from [0, 1] to [-1, 1]
    ])


    # --------------------------------------------------------
    # Datasets creation
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)

    # --------------------------------------------------------
    # DataLoaders creation
    # --------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
