import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

# determines resources allocated

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,  # path of training folder
    test_dir: str,  # path of test folder
    transform: transforms.Compose,  # a transforms object made before running this function, return to this to automate
    batch_size: int = 32,  # how many images are processed at once
    num_workers: int = NUM_WORKERS,
    subset_fraction: float = 1.0,  # determines how much of the dataset to put into dataloader
    seed: int = 42,  # fixes randomness for reproducibility
):
    def subsample_dataset(dataset, fraction, seed):
        if not (0.0 < fraction <= 1.0):
            raise ValueError("subset_fraction must be in the range (0.0, 1.0].")
        if fraction == 1.0:
            return dataset

        random.seed(seed)
        total_samples = len(dataset.samples)
        subset_size = int(fraction * total_samples)
        subset_indices = random.sample(range(total_samples), subset_size)
        dataset.samples = [dataset.samples[i] for i in subset_indices]
        dataset.targets = [dataset.targets[i] for i in subset_indices]
        return dataset

    # take in test and training directory and turns them into PyTorch dataloaders that can be interfaced by out model

    # turn image folders into datasets and apply appropriate transformations (scramble them and convert to tensors)

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_data = subsample_dataset(train_data, subset_fraction, seed)
    test_data = subsample_dataset(test_data, subset_fraction, seed)

    # turn images into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # collect labels
    class_namesfile = pd.read_csv("data/processed/labeledtest.csv", rows=1)

    class_names = class_namesfile.columns

    return train_dataloader, test_dataloader, class_names
