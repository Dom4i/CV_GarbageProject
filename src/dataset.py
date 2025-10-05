from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

from pathlib import Path
from typing import Tuple, List
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit


def get_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = None,
    val_split: float = 0.2,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Erstellt train/val DataLoader mit optionaler Augmentation und automatischer
    CPU/GPU-Optimierung (num_workers, pin_memory, prefetching).
    """

    # --- Pfad & System-Setup ---
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"{data_dir} not found"

    # dynamische Workerzahl (z. B. 6/8-Kern-CPU -> 4-6 Worker)
    if num_workers is None:
        num_workers = max(2, (os.cpu_count() or 4) - 2)

    use_cuda = torch.cuda.is_available()

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    # --- Transforms ---
    # Hier mit Parameter herumspielen
    if augment:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # --- Dataset & Stratified Split ---
    ds_full = datasets.ImageFolder(root=str(data_dir))
    y = [label for _, label in ds_full.samples]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(range(len(y)), y))

    ds_train = Subset(datasets.ImageFolder(root=str(data_dir), transform=train_tfms), train_idx)
    ds_val   = Subset(datasets.ImageFolder(root=str(data_dir), transform=val_tfms), val_idx)

    # --- DataLoader mit optimierten Parametern ---
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    train_loader = DataLoader(ds_train, shuffle=True, **loader_args)
    val_loader   = DataLoader(ds_val, shuffle=False, **loader_args)

    class_names = ds_full.classes

    print(f"✅ Dataloaders ready | "
          f"GPU={'Yes' if use_cuda else 'No'} | "
          f"Workers={num_workers} | "
          f"Batch={batch_size} | "
          f"Augment={'On' if augment else 'Off'}")

    return train_loader, val_loader, class_names


def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, targets in train_loader:
        for t in targets:
            counts[t.item()] += 1
    weights = 1.0 / (counts.float() + 1e-8)
    weights = weights / weights.sum() * num_classes  # normalisieren
    return weights

