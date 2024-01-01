"""
Contains functionality for creating PyTorch DataLoaders for image classification datasets.
"""
import os

from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """
    Create dataloaders for training and testing datasets.
    
    Args:
        train_dir (str): The directory path of the training dataset.
        test_dir (str): The directory path of the testing dataset.
        transform (torchvision.transforms.Compose): The transformation to apply to the images.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to NUM_WORKERS.
    
    Returns:
        tuple: A tuple containing the training dataloader, testing dataloader, and class names.
    """
    train_dataset = ImageFolder(root=train_dir, 
                                transform=transform,
                                target_transform=None) # Transform to perform on labels if needed

    test_dataset = ImageFolder(root=test_dir,
                               transform=transform)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_workers, # How many subprocesses to use for data loading
                                  shuffle=True,
                                  pin_memory=True) # If true, the data loader will copy Tensors into CUDA pinned memory before returning them

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers, # How many subprocesses to use for data loading
                                 shuffle=False,
                                 pin_memory=True)

    class_names = train_dataset.classes
    
    return train_dataloader, test_dataloader, class_names
    
    
