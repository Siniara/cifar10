import argparse
import json
from datetime import datetime
from time import time

import torch
import yaml
from models.tiny_vgg import TinyVGG
from torch import nn, optim
from torchmetrics.classification import Accuracy
from utils.data_loading import load_cifar10
from utils.epoch_core import test_epoch, train_epoch

torch.manual_seed(42)  # set random seed for reproducibility


def train_model(config_path: str):
    print(f"Loading config: {config_path}")
    run = config_path.split("\\")[-1].split(".")[0]
    # 1. Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Use values from config
    hidden_units = config["model"]["hidden_units"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    augmentation = config["training"]["augmentation"]
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    # DATA LOADING
    NUM_WORKERS = 2  # 0 if run on jupyter else increase
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading data")
    train_loader, validation_loader = load_cifar10(
        train=True,
        validation_split=0.2,
        return_loader=True,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    classes = train_loader.dataset.dataset.classes
    train_len = len(train_loader.dataset)
    val_len = len(validation_loader.dataset)

    # TRAINING SET UP
    model = TinyVGG(hidden_units=hidden_units, input_shape=3, output_shape=10).to(
        device
    )
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    accuracy_metric = Accuracy(task="multiclass", num_classes=len(classes)).to(device)

    MODEL_NAME = model.__class__.__name__
    MODEL_SAVE_ROOT = "checkpoints"
    MODEL_METRICS_ROOT = "metrics"

    model_filename_base = f"{MODEL_SAVE_ROOT}/{MODEL_NAME}_{run}"
    metrics_filename_base = f"{MODEL_METRICS_ROOT}/{MODEL_NAME}"
    run_id = datetime.now().strftime("%m%d%H%M")
    model_save_path = f"{model_filename_base}_{run_id}"

    best_val_loss = float("inf")
    patience_counter = 0

    # TRAINING
    model_metrics = {
        "model": MODEL_NAME,
        "device": device,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "augmentations": augmentation,
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_model_metrics = {}

    def _save_model_metrics(metrics, path: str):
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

    def _save_model(path: str):
        torch.save(model.state_dict(), path)

    print("Training model:", MODEL_NAME)
    start_time = time()
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, loss_fn, opt, accuracy_metric, device, None
        )

        val_loss, val_accuracy = test_epoch(
            model, validation_loader, loss_fn, accuracy_metric, device
        )

        model_metrics["train_loss"].append(train_loss / train_len)
        model_metrics["train_accuracy"].append(train_accuracy)
        model_metrics["val_loss"].append(val_loss / val_len)
        model_metrics["val_accuracy"].append(val_accuracy)

        # At the end of training keep track of training time
        if epoch == epochs - 1 or patience_counter >= patience:
            model_metrics["epochs"] = epoch + 1
            model_metrics["train_time"] = time() - start_time
            print(
                f"Training time on {device}: {model_metrics['train_time']:.2f} seconds"
            )

        # Training and Validation metrics for the current epoch
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / train_len:.5f},\tACC: {train_accuracy:.2f}")
        print(f"Val Loss: {val_loss / val_len:.5f},\tACC: {val_accuracy:.2f}\n")
        _save_model_metrics(
            model_metrics, f"{metrics_filename_base}_{run_id}_last.json"
        )

        # Early stopping
        if val_loss < best_val_loss:
            # Model improvement, reset counter
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model thus far
            best_model_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss / train_len,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss / val_len,
                "val_accuracy": val_accuracy,
            }
            _save_model(f"{model_save_path}_best.pth")
            _save_model_metrics(
                best_model_metrics, f"{metrics_filename_base}_{run_id}_best.json"
            )
        else:
            # No improvement, increment counter
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                _save_model(f"{model_save_path}_last.pth")
                break

    if (
        patience_counter < patience
    ):  # Save the model if not already saved by early stopping
        _save_model(f"{model_save_path}_last.pth")

    return {
        "run_id": run_id,
        "model_save_path": model_save_path,
        "best_model_metrics": best_model_metrics,
        "model_metrics": model_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    train_model(args.config)
