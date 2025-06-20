import os
from pathlib import Path
import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import json

import tqdm


NUM_WORKERS = os.cpu_count()


def subsample_dataframe(df, fraction=1.0, seed=42):
    """Subsample a pandas DataFrame by row count."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("subset_fraction must be in the range (0.0, 1.0].")
    if fraction == 1.0:
        return df

    return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)


class GeoDataset(Dataset):
    """Custom dataset to load images from CSV with hierarchical + coordinate labels."""

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = {
            "country": torch.tensor(row["country_index"], dtype=torch.long),
            "region": torch.tensor(row["region_index"], dtype=torch.long),
            "subregion": torch.tensor(row["subregion_index"], dtype=torch.long),
            "coords": torch.tensor(
                [row["latitude"], row["longitude"]], dtype=torch.float
            ),
        }

        return image, labels


def create_custom_dataloaders(
    train_csv: str,
    test_csv: str,
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int = 32,
    subset_fraction: float = 1.0,
    seed: int = 42,
):
    # Load label mappings if needed later
    with open("label_mappings.json", "r") as f:
        label_map = json.load(f)

    # Read and subsample CSVs
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = subsample_dataframe(train_df, fraction=subset_fraction, seed=seed)
    test_df = subsample_dataframe(test_df, fraction=subset_fraction, seed=seed)

    # Build custom datasets
    train_dataset = GeoDataset(train_df, image_dir=train_dir, transform=transform)
    test_dataset = GeoDataset(test_df, image_dir=test_dir, transform=transform)

    # Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Optionally return the class names if needed for visualization
    class_names = ["country", "region", "subregion", "coords"]

    return train_loader, test_loader, class_names


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    # regression_optimizer: torch.nn.Module,
    loss_fn: torch.nn.Module,
    # regression_loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    # turns a model to training mode then completes all required steps for one epoch
    model.train()

    analytics = {
        "cum_train_loss": 0,
        "cum_train_region_loss": 0,
        "cum_train_subregion_loss": 0,
        "cum_train_country_loss": 0,
        "cum_train_country_acc": 0,
        "cum_train_region_acc": 0,
        "cum_train_subregion_acc": 0,
        "cum_train_acc": 0,
    }

    # loop through dataloader batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        train_country_loss = loss_fn(y_pred["country"], y["country"])
        train_region_loss = loss_fn(y_pred["region"], y["region"])
        train_subregion_loss = loss_fn(y_pred["subregion"], y["subregion"])
        train_loss = train_country_loss + train_region_loss + train_subregion_loss

        analytics["cum_train_country_loss"] += train_country_loss.item()
        analytics["cum_train_region_loss"] += train_region_loss.item()
        analytics["cum_train_subregion_loss"] += train_subregion_loss.item()
        analytics["cum_train_loss"] += train_loss.item()

        train_loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        # calculating accuracy by converting logits (raw numbers) into propabilities and then selecting the largest probability
        y_pred_country = torch.argmax(torch.softmax(y_pred["country"], dim=1), dim=1)
        y_pred_region = torch.argmax(torch.softmax(y_pred["region"], dim=1), dim=1)
        y_pred_subregion = torch.argmax(
            torch.softmax(y_pred["subregion"], dim=1), dim=1
        )

        analytics["cum_train_country_acc"] += (
            (y_pred_country == y["country"]).sum().item()
        )
        analytics["cum_train_region_acc"] += (y_pred_region == y["region"]).sum().item()
        analytics["cum_train_subregion_acc"] += (
            (y_pred_subregion == y["subregion"]).sum().item()
        )
        analytics["cum_train_acc"] += (
            (
                y_pred_country == y["country"]
                and y_pred_region == y["region"]
                and y_pred_subregion == y["subregion"]
            )
            .sum()
            .item()
        )

    analytics = {k: v / len(dataloader) for k, v in analytics.items()}

    return analytics


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    # regression_optimizer: torch.nn.Module,
    loss_fn: torch.nn.Module,
    # regression_loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    # turns a model to eval mode then completes all required steps for one epoch
    model.eval()

    analytics = {
        "cum_test_loss": 0,
        "cum_test_region_loss": 0,
        "cum_test_subregion_loss": 0,
        "cum_test_country_loss": 0,
        "cum_test_country_acc": 0,
        "cum_test_region_acc": 0,
        "cum_test_subregion_acc": 0,
        "cum_test_acc": 0,
    }

    # loop through dataloader batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        test_country_loss = loss_fn(y_pred["country"], y["country"])
        test_region_loss = loss_fn(y_pred["region"], y["region"])
        test_subregion_loss = loss_fn(y_pred["subregion"], y["subregion"])
        test_loss = test_country_loss + test_region_loss + test_subregion_loss

        analytics["cum_test_country_loss"] += test_country_loss.item()
        analytics["cum_test_region_loss"] += test_region_loss.item()
        analytics["cum_test_subregion_loss"] += test_subregion_loss.item()
        analytics["cum_test_loss"] += test_loss.item()

        # calculating accuracy by converting logits (raw numbers) into propabilities and then selecting the largest probability
        y_pred_country = torch.argmax(torch.softmax(y_pred["country"], dim=1), dim=1)
        y_pred_region = torch.argmax(torch.softmax(y_pred["region"], dim=1), dim=1)
        y_pred_subregion = torch.argmax(
            torch.softmax(y_pred["subregion"], dim=1), dim=1
        )

        analytics["cum_test_country_acc"] += (
            (y_pred_country == y["country"]).sum().item()
        )
        analytics["cum_test_region_acc"] += (y_pred_region == y["region"]).sum().item()
        analytics["cum_test_subregion_acc"] += (
            (y_pred_subregion == y["subregion"]).sum().item()
        )
        analytics["cum_test_acc"] += (
            (
                y_pred_country == y["country"]
                and y_pred_region == y["region"]
                and y_pred_subregion == y["subregion"]
            )
            .sum()
            .item()
        )

    analytics = {k: v / len(dataloader) for k, v in analytics.items()}

    return analytics


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    test_frequency: int,
) -> Dict[str, List]:
    # pass a target model through train and test steps for x epochs. Calculate and store eval metrics

    # create an empty dictionary to send out later (intentionally leaving off coord stats for staged learning to not overwhelm system)
    results = {
        "cum_train_loss": [],
        "cum_train_acc": [],
        "cum_train_country_loss": [],
        "cum_train_country_acc": [],
        "cum_train_region_loss": [],
        "cum_train_region_acc": [],
        "cum_train_subregion_loss": [],
        "cum_train_subregion_acc": [],
        # "cum_train_coords_loss": [],
        # "cum_train_coords_acc": [],
        "cum_test_loss": [],
        "cum_test_acc": [],
        "cum_test_country_loss": [],
        "cum_test_country_acc": [],
        "cum_test_region_loss": [],
        "cum_test_region_acc": [],
        "cum_test_subregion_loss": [],
        "cum_test_subregion_acc": [],
        # "cum_test_coords_loss": [],
        # "cum_test_coords_acc": [],
    }

    for epoch in tqdm(range(epochs)):  # explain
        train_metrics = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        for key, value in train_metrics.items():
            results[key].append(value)

        if epoch % test_frequency == 0:
            test_metrics = test_step(
                model=model,
                dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )

            for key, value in test_metrics.items():
                results[key].append(value)
        print(f"epoch: {epoch + 1}train_loss: {results[epoch]}")  # explain

    return results


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), (
        "model_name must end with '.pt' or '.pth'"
    )

    model_save_path = target_dir_path / model_name

    print(f"[INFO] saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
