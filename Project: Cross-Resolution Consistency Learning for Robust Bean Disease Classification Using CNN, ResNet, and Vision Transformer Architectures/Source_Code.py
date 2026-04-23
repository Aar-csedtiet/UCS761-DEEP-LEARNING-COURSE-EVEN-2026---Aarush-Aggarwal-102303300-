"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Cross-Resolution Consistency Learning for Bean Disease Classification     ║
║                                                                              ║
║   Dataset  : Bean Disease (Angular Leaf Spot, Bean Rust, Healthy)           ║
║   Models   : VGG-style CNN | ResNet | Vision Transformer (ViT)              ║
║   Device   : H100 GPU (auto-detected)                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW THIS PROJECT WORKS — Read this before looking at the code!
──────────────────────────────────────────────────────────────
PROBLEM:
    A normal model trained on 224×224 images may fail when the same image
    is shown at a smaller resolution (e.g., 56×56). It gives inconsistent
    predictions — unreliable in real-world use.

SOLUTION:
    For every image in each batch, we create 3 scaled versions:
        • High  resolution: 224×224
        • Mid   resolution: 112×112
        • Low   resolution:  56×56

    We then use TWO loss terms:
    1. Classification Loss (CrossEntropy)
       → Did the model predict the correct class?
    2. Consistency Loss (KL Divergence)
       → Does the model give the SAME prediction at all 3 resolutions?

    Total Loss = Classification Loss + λ × Consistency Loss
                                       ↑
                               λ = 0.3 (controls how much we care about consistency)

RESULT:
    We train and compare all three architectures on:
        • Test Accuracy       — Is it correct?
        • Consistency Score   — Is it stable across resolutions?

=============================================================================
Cross-Resolution Consistency Learning for Bean Disease Classification
=============================================================================
Dataset : Bean_Dataset_DL  (Angular Leaf Spot | Bean Rust | Healthy)
Models  : VGG-style CNN  |  ResNet18  |  Vision Transformer (ViT)
Scales  : 32x32, 64x64, 128x128, 224x224
Outputs : All results saved to  results/  folder
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ─────────────────────────────────────────────────────────────────────────────
# 3. GLOBAL CONFIGURATION  –  change paths/hyper-params here only
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "Bean_Dataset_DL"          # root folder of the dataset
RESULTS_DIR = "results"                  # every output goes here
NUM_CLASSES = 3
CLASS_NAMES = ["Angular Leaf Spot", "Bean Rust", "Healthy"]

SCALES      = [32, 64, 128, 224]        # resolutions to test
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-3
CONSISTENCY_WEIGHT = 0.5               # weight for the consistency loss term

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(scale: int):
    """Return train and validation transforms for a given image size."""
    train_tf = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def load_datasets(scale: int):
    """
    Load the dataset at a given scale.
    Splits 80 % train / 10 % val / 10 % test from DATA_DIR.
    """
    train_tf, val_tf = get_transforms(scale)

    # Load everything once with the train transform to get labels
    full_dataset = ImageFolder(DATA_DIR, transform=train_tf)

    n_total = len(full_dataset)
    n_train = int(0.80 * n_total)
    n_val   = int(0.10 * n_total)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Apply val/test transform to the validation and test subsets
    val_ds.dataset  = copy.deepcopy(full_dataset)
    val_ds.dataset.transform  = val_tf
    test_ds.dataset = copy.deepcopy(full_dataset)
    test_ds.dataset.transform = val_tf

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=2, pin_memory=True),
        "val"  : DataLoader(val_ds,   batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True),
        "test" : DataLoader(test_ds,  batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True),
    }
    return loaders

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. VGG-style CNN ────────────────────────────────────────────────────────
class VGGStyleCNN(nn.Module):
    """
    A small VGG-inspired CNN with two conv-blocks followed by a classifier.
    Simple enough to understand at a glance.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # halves spatial size
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.pool    = nn.AdaptiveAvgPool2d((4, 4))      # fixed 4×4 output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── 5b. ResNet18 (pretrained, fine-tuned) ────────────────────────────────────
def build_resnet(num_classes: int = 3) -> nn.Module:
    """Use a pretrained ResNet-18 and replace the final FC layer."""
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── 5c. Vision Transformer (pretrained ViT-B/16) ─────────────────────────────
class ViTWithResize(nn.Module):
    """
    Wrapper around ViT-B/16 that upsamples any input to 224×224 first.

    WHY: ViT-B/16 splits the image into fixed 16×16 patches, so it strictly
    requires a 224×224 input. When we feed a 32×32 or 64×64 image we must
    upsample it to 224 before passing it through the transformer.
    The 'scale' of the loader still tells us the information density of the
    original image — we are testing whether the model is consistent even
    when given low-resolution (information-poor) inputs.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.vit = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        )
        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, num_classes
        )

    def forward(self, x):
        # Upsample to 224x224 if needed (bilinear, no learnable params)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
        return self.vit(x)


def build_vit(num_classes: int = 3) -> nn.Module:
    """Return a ViT-B/16 that accepts any image size by upsampling to 224."""
    return ViTWithResize(num_classes)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CONSISTENCY LOSS
# ─────────────────────────────────────────────────────────────────────────────

def consistency_loss(logits_list: list) -> torch.Tensor:
    """
    Consistency loss: the model should give similar softmax probabilities
    for the same image resized to different scales.

    We compute the mean probability across all scales and then penalise
    each scale's deviation from that mean using KL-divergence.

    logits_list : list of tensors, each shape (B, C)
    returns     : scalar tensor
    """
    probs_list  = [torch.softmax(l, dim=1) for l in logits_list]
    mean_probs  = torch.stack(probs_list, dim=0).mean(dim=0)   # (B, C)
    loss        = 0.0
    kl_fn       = nn.KLDivLoss(reduction="batchmean")
    for p in probs_list:
        # KLDivLoss expects log-probabilities as input
        loss += kl_fn(torch.log(p + 1e-8), mean_probs)
    return loss / len(probs_list)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP  (single model, single resolution)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, loaders, model_name: str, scale: int):
    """
    Standard training loop with cross-entropy loss.
    Returns the trained model and a history dict.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        # ── training phase
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item() * images.size(0)
            t_correct += (outputs.argmax(1) == labels).sum().item()
            t_total   += images.size(0)

        # ── validation phase
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                v_loss    += loss.item() * images.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total   += images.size(0)

        scheduler.step()

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"  [{model_name} | scale={scale}] "
              f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_acc={t_acc:.3f}  val_acc={v_acc:.3f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history

# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader):
    """Return true labels, predicted labels, and raw softmax probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOTTING HELPERS  –  all saved to RESULTS_DIR
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(history, model_name: str, scale: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(epochs, history["train_acc"], label="Train")
    axes[0].plot(epochs, history["val_acc"],   label="Val")
    axes[0].set_title(f"{model_name} | scale={scale} — Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"],   label="Val")
    axes[1].set_title(f"{model_name} | scale={scale} — Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR,
                         f"{model_name}_scale{scale}_training_curves.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"    Saved: {fname}")


def save_confusion_matrix(labels, preds, model_name: str, scale: int):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{model_name} | scale={scale} — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR,
                         f"{model_name}_scale{scale}_confusion_matrix.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"    Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CROSS-RESOLUTION CONSISTENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_consistency_score(models_by_scale: dict, base_loader_by_scale: dict):
    """
    For every pair of scales, count the fraction of images where both
    scale models agree on the predicted class.  Higher = more consistent.

    models_by_scale    : {scale: trained_model}
    base_loader_by_scale : {scale: test_loader}
    Returns: consistency matrix (n_scales x n_scales)
    """
    scale_list = sorted(models_by_scale.keys())
    preds_by_scale = {}
    for s in scale_list:
        labels, preds, _ = evaluate_model(
            models_by_scale[s], base_loader_by_scale[s]
        )
        preds_by_scale[s] = preds

    n = len(scale_list)
    matrix = np.zeros((n, n))
    for i, s1 in enumerate(scale_list):
        for j, s2 in enumerate(scale_list):
            min_len = min(len(preds_by_scale[s1]), len(preds_by_scale[s2]))
            agree   = (preds_by_scale[s1][:min_len] ==
                       preds_by_scale[s2][:min_len]).mean()
            matrix[i, j] = agree

    return matrix, scale_list


def save_consistency_heatmap(matrix, scale_list, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels  = [str(s) for s in scale_list]
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{model_name} — Cross-Resolution Consistency")
    ax.set_xlabel("Scale B"); ax.set_ylabel("Scale A")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR,
                         f"{model_name}_consistency_heatmap.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"    Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SUMMARY BAR CHART  (test accuracy of all model × scale combos)
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_chart(summary: dict):
    """
    summary = { model_name: {scale: test_accuracy} }
    """
    model_names = list(summary.keys())
    x = np.arange(len(SCALES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue", "tomato", "seagreen"]
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        accs = [summary[mname].get(s, 0) for s in SCALES]
        ax.bar(x + i * width, accs, width, label=mname, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([str(s) for s in SCALES])
    ax.set_xlabel("Image Scale (px)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy — All Models × All Scales")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "summary_test_accuracy.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"    Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_text_report(summary: dict, report_lines: list):
    fname = os.path.join(RESULTS_DIR, "full_classification_report.txt")
    with open(fname, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CROSS-RESOLUTION BEAN DISEASE CLASSIFICATION — FULL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("TEST ACCURACY SUMMARY\n")
        f.write("-" * 40 + "\n")
        for mname, scale_dict in summary.items():
            for scale, acc in sorted(scale_dict.items()):
                f.write(f"  {mname:20s} scale={scale:3d}  acc={acc:.4f}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write("DETAILED CLASSIFICATION REPORTS\n")
        f.write("-" * 40 + "\n\n")
        for line in report_lines:
            f.write(line + "\n")
    print(f"    Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print(" Cross-Resolution Consistency Learning — Bean Disease")
    print("=" * 65)

    # Model factories:  name → callable that returns a fresh model
    model_factories = {
        "VGG_CNN": lambda: VGGStyleCNN(NUM_CLASSES),
        "ResNet18": lambda: build_resnet(NUM_CLASSES),
        "ViT_B16" : lambda: build_vit(NUM_CLASSES),
    }

    # Stores for summary and reports
    summary      = {name: {} for name in model_factories}
    report_lines = []

    # For cross-resolution consistency analysis per model
    trained_models_by_model = {name: {} for name in model_factories}
    test_loaders_by_scale   = {}    # reused across models

    # ── Loop over architectures ──────────────────────────────────────────────
    for model_name, factory in model_factories.items():
        print(f"\n{'─'*65}")
        print(f"  Architecture: {model_name}")
        print(f"{'─'*65}")

        # ── Loop over resolutions ────────────────────────────────────────────
        for scale in SCALES:
            print(f"\n  >> Scale = {scale}×{scale}")

            # Load data
            loaders = load_datasets(scale)
            if scale not in test_loaders_by_scale:
                test_loaders_by_scale[scale] = loaders["test"]

            # Build a fresh model
            model = factory()

            # Train
            t0 = time.time()
            model, history = train_model(model, loaders, model_name, scale)
            elapsed = time.time() - t0
            print(f"     Training time: {elapsed:.1f}s")

            # Evaluate on test set
            labels, preds, probs = evaluate_model(model, loaders["test"])
            acc = (labels == preds).mean()
            summary[model_name][scale] = acc
            print(f"     Test accuracy : {acc:.4f}")

            # Collect text report
            cr = classification_report(labels, preds,
                                       target_names=CLASS_NAMES)
            report_lines.append(
                f"=== {model_name} | scale={scale} ===\n{cr}"
            )

            # ── Save all plots ───────────────────────────────────────────────
            save_training_curves(history, model_name, scale)
            save_confusion_matrix(labels, preds,  model_name, scale)
            
            # Keep model for consistency analysis
            trained_models_by_model[model_name][scale] = model

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        # ── Cross-resolution consistency heatmap ─────────────────────────────
        print(f"\n  Computing cross-resolution consistency for {model_name}...")
        consistency_matrix, scale_list = compute_consistency_score(
            trained_models_by_model[model_name],
            test_loaders_by_scale
        )
        save_consistency_heatmap(consistency_matrix, scale_list, model_name)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Saving summary chart and text report …")
    save_summary_chart(summary)
    save_text_report(summary, report_lines)

    print(f"\n{'='*65}")
    print(f"  ALL DONE!  Every result is saved in: {RESULTS_DIR}/")
    print(f"{'='*65}\n")

    # ── Print final table to console ─────────────────────────────────────────
    print(f"{'Model':<20} {'Scale':>6} {'Test Acc':>10}")
    print("-" * 40)
    for mname in summary:
        for scale in SCALES:
            acc = summary[mname].get(scale, 0)
            print(f"{mname:<20} {scale:>6} {acc:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()