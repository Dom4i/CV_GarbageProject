import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----- 1. Loss & Accuracy Plot -----
def plot_training_history(history: dict):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss per Epoch")

    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend(); plt.title("Accuracy per Epoch")
    plt.tight_layout()
    plt.show()


# ----- 2. Confusion Matrix Heatmap -----
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ----- 3. Klassenverteilung -----
def plot_class_distribution(labels, class_names):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8,4))
    plt.bar(class_names, counts, color="skyblue")
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()


def plot_error_rate_per_class(all_targets, all_preds, class_names):
    class_errors = []
    class_totals = []
    for i, name in enumerate(class_names):
        errors = sum(1 for t, p in zip(all_targets, all_preds) if t == i and p != i)
        total = sum(1 for t in all_targets if t == i)
        class_errors.append(errors)
        class_totals.append(total)

    class_error_rate = [e/t if t>0 else 0 for e,t in zip(class_errors, class_totals)]

    plt.figure(figsize=(8,4))
    plt.bar(class_names, class_error_rate, color="salmon")
    plt.ylabel("Error Rate")
    plt.title("Validation Error Rate per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_prediction_confidence(probs, targets, class_names):
    plt.figure(figsize=(10,5))
    n_classes = len(class_names)

    # probs: List of lists von softmax-Wahrscheinlichkeiten
    # targets: List von Integer-Labels
    for i, class_name in enumerate(class_names):
        class_probs = [p[i] for p, t in zip(probs, targets) if t == i]
        plt.hist(class_probs, bins=20, alpha=0.5, label=class_name)

    plt.xlabel("Predicted probability for true class")
    plt.ylabel("Number of samples")
    plt.title("Prediction Confidence Distribution per Class")
    plt.legend()
    plt.tight_layout()
    plt.show()


