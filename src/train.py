from pathlib import Path
from typing import Dict, Tuple
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from .dataset import get_dataloaders, compute_class_weights
from .model_cnn import SmallCNN
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # für schnellere Trainingsläufe
from torch import amp as torch_amp  # <-- einmalig ganz oben in train.py

def seed_everything(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """
    Trainiert genau eine Epoche.
    Nutzt automatische Mixed Precision (AMP), wenn:
      - device 'cuda' ist und
      - ein GradScaler übergeben wurde.
    """
    use_cuda = (device.type == "cuda")
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=use_cuda)
        yb = yb.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and use_cuda:
            # AMP (neue API)
            with torch_amp.autocast("cuda", enabled=True):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32-Fallback (CPU oder ohne Scaler)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, Dict[str, int]]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(yb.cpu().tolist())
    return (running_loss / total,
            correct / total,
            {"preds": all_preds, "targets": all_targets})
def fit(
    data_dir: str = "data/TrashType_Image_Dataset",
    models_dir: str = "models",
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 1e-3,
    val_split: float = 0.2,
    use_class_weights: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 5,   # None = deaktivieren
):
    import json

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=val_split,
        augment=True,
    )
    num_classes = len(class_names)

    # Model & loss
    model = SmallCNN(num_classes=num_classes).to(device)

    if use_class_weights:
        w = compute_class_weights(train_loader, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1) # Label Smoothing hilft bei Klassen, die sich visuell ähneln (glass/metal/plastic).
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch_amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Paths
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    best_weights = models_path / "best_weights.pt"
    classes_json = models_path / "classes.json"

    # Training loop
    best_val = float("inf")
    no_improve = 0

    # Klassenliste separat speichern (praktisch für spätere Nutzung)
    with open(classes_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | {dt:.1f}s"
        )

        # Save best weights only
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_weights)
        else:
            no_improve += 1

        # Early stopping
        if early_stopping_patience is not None and no_improve >= early_stopping_patience:
            print(f"Early stopping (no val_loss improvement for {early_stopping_patience} epochs).")
            break

    # ----- Abschlussreport mit sicherem Laden -----
    # (nur Gewichte, keine Pickle-Objekte)
    state_dict = torch.load(best_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # Klassenliste laden (falls extern gebraucht)
    try:
        with open(classes_json, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception:
        pass  # falls Datei gelöscht wurde, bleibt class_names aus dem Speicher

    _, _, out = evaluate(model, val_loader, criterion, device)
    print("\nClassification report (val):")
    print(classification_report(out["targets"], out["preds"], target_names=class_names))
    print("Confusion matrix (val):")
    print(confusion_matrix(out["targets"], out["preds"]))
    print(f"\nBest checkpoint saved to: {best_weights.resolve()}")
