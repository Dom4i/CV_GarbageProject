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
    batch_size: int = 32, # Wie viele Bilder pro Batch geladen werden -> pro Forward-Pass werden 32 Bilder verarbeitet
    num_workers: int = None, # Anzahl der parallelen Worker
    val_split: float = 0.2,
    augment: bool = True,
    preview: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str]]: # Funktion gibt Tupel (Mehrfach Rückgabewert) zurück aus: DataLoader Training, DataLoader Validation und Liste mit Klassennamen
    """
    Erstellt train/val DataLoader mit optionaler Augmentation und automatischer
    CPU/GPU-Optimierung (num_workers, pin_memory, prefetching).
    """

    # --- Pfad & System-Setup ---
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"{data_dir} not found"

    # dynamische Workerzahl (z. B. 6/8-Kern-CPU -> 4-6 Worker)
    if num_workers is None:
        num_workers = max(2, (os.cpu_count() or 4) - 2) # nimmt logische Prozessoren -2 (damit paar Threads für andere Prozesse bleiben)

    # Standardwerte zum Normalisieren von Bildern (so wird es häufig in PyTorch verwendet), für pretrained Modelle notwendig aber auch für neu trainierte Modelle verwendbar (für benchmarking sinnvoll)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    # --- Transforms ---
    # Hier mit Augmentation Parameter herumspielen:
    if augment: # augmentation wird für jede trainings epoche neu durchgeführt -> nachfolgend sind random augmentation, d.h. in jedem epoch sind leicht unterschiedliche Bilder als Trainingsdaten
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else: # keine Augmentation
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
    ds_full = datasets.ImageFolder(root=str(data_dir)) # gesamten Datensatz importieren -> Funktion erkennt Unterverzeichnisse automatisch und wird daher als eigene Klasse interpretiert
    y = [label for _, label in ds_full.samples] # Klassenlabels extrahieren: ds_full.samples ist Liste von (Dateipfad, Label)-Tupel; Beispiel: [("data/paper/img001.jpg", 0), ("data/plastic/img034.jpg", 1), ...]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42) # mit stratified werden Klassenanteile in Training und Validierung etwa gleich groß
    train_idx, val_idx = next(splitter.split(range(len(y)), y)) # führt den Split aus -> erzeugt Index Listen für Train und Validation

    # Erstelle Teil-Datasets (Subsets) auf Basis der Index-Listen
    ds_train = Subset(datasets.ImageFolder(root=str(data_dir), transform=train_tfms), train_idx)
    ds_val   = Subset(datasets.ImageFolder(root=str(data_dir), transform=val_tfms), val_idx)

    use_cuda = torch.cuda.is_available()

    # --- DataLoader mit optimierten Parametern ---
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0), # Behalte die Worker-Prozesse über Epochen hinweg am Leben -> Zeitersparnis zwischen den Epochen
        prefetch_factor=4 if num_workers > 0 else None, # Jeder Worker lädt im Voraus 4 Batches
    )

    # DataLoader ist PyTorch Objekt, das Bilder batchweise aus dem Datensatz lädt:
    train_loader = DataLoader(ds_train, shuffle=True, **loader_args) # mit shuffle=True wird Reihenfolge der Bilder vermischt, sodass Modell nicht die Reihenfolge antrainiert statt der Bild Features
    val_loader   = DataLoader(ds_val, shuffle=False, **loader_args)

    class_names = ds_full.classes

    print(f"Dataloaders ready | "
          f"GPU={'Yes' if use_cuda else 'No'} | "
          f"Workers={num_workers} | "
          f"Batch={batch_size} | "
          f"Augment={'On' if augment else 'Off'}")

    # Optional: Beispielbilder anzeigen
    if preview:
        import matplotlib.pyplot as plt
        import numpy as np

        xb, yb = next(iter(train_loader))
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        # Hole die zugrunde liegende ImageFolder-Instanz
        base_dataset = ds_train.dataset
        # Verwende die train_idx-Liste (Subset weiß, welche Indizes verwendet wurden)
        subset_indices = ds_train.indices

        # Wir zeigen dieselben Bilder wie im ersten Batch an
        n_show = min(12, xb.size(0))
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.flatten()

        for i in range(n_show):
            img = xb[i].cpu() * std + mean  # Rücknormalisierung
            img = img.permute(1, 2, 0).numpy()
            path, _ = base_dataset.samples[subset_indices[i]]  # Dateipfad aus Original-Dataset
            filename = os.path.basename(path)

            axes[i].imshow(np.clip(img, 0, 1))
            axes[i].set_title(
                f"{class_names[yb[i].item()]}\n({filename})",
                fontsize=9
            )
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    return train_loader, val_loader, class_names


def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, targets in train_loader:
        for t in targets:
            counts[t.item()] += 1
    weights = 1.0 / (counts.float() + 1e-8)
    weights = weights / weights.sum() * num_classes  # normalisieren
    return weights

