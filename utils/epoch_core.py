"""
Contains functions to:
- Run a single epoch of training or testing (`run_epoch`)
- Train or test for one epoch (`train_epoch`, `test_epoch`)
- Process a single batch (`train_batch`, `predict_batch`)
- Compute batch metrics (`evaluate_batch`)
"""

from typing import Callable, Generator

import torch
import torchmetrics
from torch import nn
from tqdm import tqdm


def run_epoch(
    model: nn.Module,
    data: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_step_fn: Callable,
    optimiser: torch.optim.Optimizer,
    accuracy_metric: torchmetrics.Metric,
    device: str,
) -> Generator[tuple[int, float, torch.Tensor, float, float], None, None]:
    """
    Core logic for running one epoch of training or testing.

    Args:
    - loss_fn: A PyTorch loss function (e.g., `torch.nn.CrossEntropyLoss()`)
    - batch_step_fn: Callable that processes a single batch (e.g. train_batch or predict_batch). Must return (logits, loss).

    Yields
    - batch_idx: int
    - batch_loss: float
    - class_preds: torch.Tensor
    - batch_accuracy: float
    - cumulative_accuracy: float
    """
    accuracy_metric.reset()

    for batch_idx, (features, labels) in tqdm(enumerate(data), total=len(data)):
        features = features.to(device)
        labels = labels.to(device)

        logits, batch_loss = batch_step_fn(model, features, labels, loss_fn, optimiser)
        class_preds, batch_accuracy = evaluate_batch(logits, labels, accuracy_metric)

        yield (
            batch_idx,
            batch_loss.item(),
            class_preds,
            batch_accuracy.item(),
            accuracy_metric.compute().item(),
        )


def train_epoch(
    model: nn.Module,
    data: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimiser: torch.optim.Optimizer,
    accuracy_metric: torchmetrics.Metric,
    device: str,
    print_interval: int = None,
    return_batch_metrics: bool = False,
) -> (
    tuple[float, float]  # cumulative loss + cumulative accuracy
    | tuple[float, float, list[float], list[float]]  # + batch losses + batch accuracies
):
    """Train the model for one epoch. Wrapper around run_epoch. Accumulates loss and accuracy.
    Args:
    - loss_fn: A PyTorch loss function (e.g., `torch.nn.CrossEntropyLoss()`)
    - print_interval: If provided, prints cumulative loss and accuracy every `print_interval` batches.
    - return_batch_metrics: If True, returns lists of batch losses and accuracies. For batch-level analysis.
    Returns:
    - average_loss: Average loss over the epoch.
    - train_accs: List of batch accuracies.
    - cumulative_accuracy: Cumulative accuracy over the epoch."""
    train_loss = 0
    train_losses = [] if return_batch_metrics else None
    train_accs = [] if return_batch_metrics else None

    for (
        batch_idx,
        batch_train_loss,
        _,
        batch_train_accuracy,
        cumulative_train_accuracy,
    ) in run_epoch(
        model=model,
        data=data,
        loss_fn=loss_fn,
        batch_step_fn=train_batch,
        optimiser=optimiser,
        accuracy_metric=accuracy_metric,
        device=device,
    ):
        train_loss += batch_train_loss
        if return_batch_metrics:
            train_losses.append(batch_train_loss)
            train_accs.append(batch_train_accuracy)

        if print_interval is not None:
            if batch_idx == len(data) - 1 or batch_idx % print_interval == 0:
                print(
                    f"Batch {batch_idx} |\t Cumulative Loss: {train_loss:.4f},\tCumulative Accuracy: {cumulative_train_accuracy:.2f}"
                )

    output = train_loss, cumulative_train_accuracy
    if return_batch_metrics:
        output += (
            train_loss,
            train_accs,
        )

    return output


def test_epoch(
    model: nn.Module,
    data: torch.utils.data.DataLoader,
    loss_fn: Callable,
    accuracy_metric: torchmetrics.Metric,
    device: str,
    return_batch_metrics: bool = False,
    return_preds: bool = False,
) -> (
    tuple[float, float]  # only loss + cumulative accuracy
    | tuple[float, float, list[float], list[float]]  # + batch losses + batch accuracies
    | tuple[float, float, list[float], list[float], list[torch.Tensor]]  # + preds
    | tuple[float, float, list[torch.Tensor]]  # only preds without batch metrics
):
    """Test the model for one epoch. Wrapper around run_epoch. Accumulates loss and accuracy.
    Args:
    - loss_fn: A PyTorch loss function (e.g., `torch.nn.CrossEntropyLoss()`)
    - return_batch_metrics: If True, returns lists of batch losses and accuracies. For batch-level analysis.
    - return_preds: If True, returns all predictions made during the epoch.
    Returns:
    - average_loss: Average loss over the epoch.
    - test_accs: List of batch accuracies.
    - cumulative_accuracy: Cumulative accuracy over the epoch.
    - preds (optional): List of all predictions made during the epoch."""
    test_loss = 0
    test_losses = [] if return_batch_metrics else None
    test_accs = [] if return_batch_metrics else None
    preds = [] if return_preds else None

    for (
        _,
        batch_test_loss,
        batch_preds,
        batch_test_accuracy,
        cumulative_test_accuracy,
    ) in run_epoch(
        model=model,
        data=data,
        loss_fn=loss_fn,
        batch_step_fn=_predict_batch_wrapper,
        optimiser=None,
        accuracy_metric=accuracy_metric,
        device=device,
    ):
        test_loss += batch_test_loss
        if return_batch_metrics:
            test_losses.append(batch_test_loss)
            test_accs.append(batch_test_accuracy)
        if return_preds:
            preds.extend(batch_preds)

    output = test_loss, cumulative_test_accuracy
    if return_batch_metrics:
        output += (test_losses, test_accs)
    if return_preds:
        output += (preds,)

    return output


def train_batch(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable,
    optimiser: torch.optim.Optimizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train the model on a single batch of data.
    Args:
    - loss_fn: A PyTorch loss function (e.g., `torch.nn.CrossEntropyLoss()`)
    Returns:
    - logits: The raw output from the model before applying activation functions.
    - loss: The computed loss for the batch.
    """
    model.train()

    logits = model(features)
    loss = loss_fn(input=logits, target=target)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return logits, loss


def predict_batch(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass the model on a single batch of data for prediction. No gradient tracking. Used for testing.
    Args:
    - loss_fn: A PyTorch loss function (e.g., `torch.nn.CrossEntropyLoss()`)
    Returns:
    - logits: The raw output from the model before applying activation functions.
    - loss: The computed loss for the batch.
    """
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        loss = loss_fn(input=logits, target=target)

    return logits, loss


def _predict_batch_wrapper(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable,
    optimiser=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper to match the signature of train_batch for use in run_epoch. Ignores the optimiser argument."""
    return predict_batch(model, features, target, loss_fn)


def evaluate_batch(
    raw_predictions, labels, accuracy_metric
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes batch predictions and evaluates accuracy using the provided accuracy_metric."""
    class_preds = raw_predictions.argmax(dim=1)
    accuracy = accuracy_metric(preds=class_preds, target=labels)
    return class_preds, accuracy
