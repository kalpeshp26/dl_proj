"""
DeepRetail — MobileNetV3-Small Classifier Training (Stage 3)

Architecture:
    torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    Final layer replaced with nn.Linear(in_features, num_classes)

Training protocol:
    Phase A — freeze backbone, train head for CLASSIFIER_EPOCHS_FROZEN epochs
    Phase B — unfreeze all, fine-tune for CLASSIFIER_EPOCHS_FINETUNE epochs

Outputs:
    models/classifier_best.pt   — full model state dict
    logs/classifier_eval.json   — validation accuracy + confusion matrix

Usage:
    python src/classification/train_classifier.py
    python src/classification/train_classifier.py --epochs-frozen 5 --epochs-finetune 20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CLASSIFIER_DATASET_DIR, MODELS_DIR, CLASSIFIER_MODEL_PATH,
    PRODUCT_CLASSES, CLASSIFIER_IMG_SIZE, CLASSIFIER_BATCH,
    CLASSIFIER_EPOCHS_FROZEN, CLASSIFIER_EPOCHS_FINETUNE,
    CLASSIFIER_LR_HEAD, CLASSIFIER_LR_FINETUNE, CLASSIFIER_WEIGHT_DECAY,
    CLASSIFIER_VAL_SPLIT, LOG_DIRECTORY,
)


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── Model builder ───────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """Load pretrained MobileNetV3-Small, replace the classifier head."""
    import torchvision.models as models
    backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Replace final linear layer
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    return total_loss / total, correct / total, confusion


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MobileNetV3-Small product classifier")
    p.add_argument("--epochs-frozen", type=int, default=CLASSIFIER_EPOCHS_FROZEN)
    p.add_argument("--epochs-finetune", type=int, default=CLASSIFIER_EPOCHS_FINETUNE)
    p.add_argument("--batch", type=int, default=CLASSIFIER_BATCH)
    p.add_argument("--lr-head", type=float, default=CLASSIFIER_LR_HEAD)
    p.add_argument("--lr-finetune", type=float, default=CLASSIFIER_LR_FINETUNE)
    p.add_argument("--device", type=str, default="",
                   help="'cpu', 'cuda', 'mps' or '' for auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Dataset ──────────────────────────────────────────────────
    if not CLASSIFIER_DATASET_DIR.exists() or not any(CLASSIFIER_DATASET_DIR.iterdir()):
        raise FileNotFoundError(
            f"Classifier dataset not found at {CLASSIFIER_DATASET_DIR}\n"
            "Create subdirectories per product class:\n"
            "  data/classifier_dataset/<class_name>/*.jpg\n"
            "Run: python src/data_tools/capture_images.py --all  to collect images."
        )

    full_dataset = ImageFolder(
        str(CLASSIFIER_DATASET_DIR),
        transform=get_train_transforms(CLASSIFIER_IMG_SIZE),
    )
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(full_dataset.classes)
    print(f"[Classifier] Classes ({num_classes}): {full_dataset.classes}")
    print(f"[Classifier] Total images: {len(full_dataset)}")

    # Train/val split
    val_size = max(1, int(len(full_dataset) * CLASSIFIER_VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Val uses different transforms
    val_ds.dataset.transform = get_val_transforms(CLASSIFIER_IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Device ───────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Classifier] Device: {device}")

    # ── Model ────────────────────────────────────────────────────
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    # ── Phase A: frozen backbone ─────────────────────────────────
    freeze_backbone(model)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head,
        weight_decay=CLASSIFIER_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_frozen
    )

    print(f"\n[Phase A] Training head only — {args.epochs_frozen} epochs")
    for epoch in range(1, args.epochs_frozen + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs_frozen} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"{elapsed:.1f}s")
        history.append({"phase": "A", "epoch": epoch,
                        "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CLASSIFIER_MODEL_PATH)

    # ── Phase B: full fine-tune ───────────────────────────────────
    unfreeze_all(model)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr_finetune,
        weight_decay=CLASSIFIER_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_finetune
    )

    print(f"\n[Phase B] Full fine-tune — {args.epochs_finetune} epochs")
    for epoch in range(1, args.epochs_finetune + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, confusion = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs_finetune} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"{elapsed:.1f}s")
        history.append({"phase": "B", "epoch": epoch,
                        "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CLASSIFIER_MODEL_PATH)
            print(f"  [Saved] New best val_acc={val_acc:.4f}")

    # ── Final evaluation + confusion matrix ──────────────────────
    print(f"\n[Classifier] Best val_acc = {best_val_acc:.4f}")
    print(f"[Classifier] Weights → {CLASSIFIER_MODEL_PATH}")

    # Load best weights for final eval
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    _, final_acc, confusion = evaluate(model, val_loader, criterion, device, num_classes)

    eval_report = {
        "val_accuracy": round(final_acc, 4),
        "num_classes": num_classes,
        "classes": full_dataset.classes,
        "class_to_idx": class_to_idx,
        "confusion_matrix": confusion.tolist(),
        "training_history": history,
    }
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    eval_path = LOG_DIRECTORY / "classifier_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[Classifier] Eval report → {eval_path}")

    # Print confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "         " + "  ".join(f"{c[:8]:>8}" for c in full_dataset.classes)
    print(header)
    for i, row in enumerate(confusion):
        lbl = full_dataset.classes[i][:8]
        print(f"{lbl:>8}   " + "  ".join(f"{v:>8}" for v in row))

    # Save class index mapping alongside model
    idx_path = MODELS_DIR / "class_indices.json"
    with open(idx_path, "w") as f:
        json.dump({"class_to_idx": class_to_idx,
                   "idx_to_class": {str(v): k for k, v in class_to_idx.items()}}, f, indent=2)
    print(f"[Classifier] Class index map → {idx_path}")


if __name__ == "__main__":
    main()
