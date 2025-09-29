import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


def load_cifar10(
    train: bool,
    raw: bool = False,
    augmentation: bool = False,
    validation_split: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 2,
    return_loader: bool = True,
) -> DataLoader | Dataset | tuple[DataLoader, DataLoader] | tuple[Dataset, Dataset]:
    """
    Load the CIFAR-10 dataset. Can return either DataLoader objects or raw datasets for data exploration.

    Returns:
    - If train=True: (trainset, evalset) → trainset for training, evalset for validation or test.
    - If train=False: evalset only → the official test set
    - If return_loader=True, returns DataLoader objects instead of datasets.

    Load the CIFAR-10 dataset for training or evaluation.

    Features:
    - Returns either DataLoader objects (default) or raw Dataset objects.
    - Supports train/validation split.
    - Optional data augmentation for training.
    - Normalizes images to [0,1] and standardizes to [-1, 1] with (mean/std of 0.5).

    Parameters:
    ----------
    train : If True, load the training set (with optional validation split). If False, load the test set.
    raw : If True, skips the train/test split logic and just returns the dataset. Useful for data exploration.
    augmentation : If True and train=True, applies random horizontal flip and random crop.
    validation_split : Fraction of training data to use as validation set (0 to 1). Only used if train=True.
    batch_size : Number of samples per batch if returning DataLoaders.
    num_workers : int, default=2 Number of worker processes for DataLoader. !Must be 0 for notebooks.!
    return_loader : If True, returns DataLoader(s); if False, returns Dataset(s).

    Returns:
    -------
    If train=True:
        Tuple[DataLoader, DataLoader] or Tuple[Dataset, Dataset] → (train, validation/test)
    If train=False:
        DataLoader or Dataset → test set only

    Example usage:
    --------------
    # Get training and validation loaders with augmentation
    train_loader, val_loader = load_cifar10(train=True, augmentation=True, validation_split=0.1)

    # Get raw training dataset without augmentation (useful for data exploration)
    train_dataset = load_cifar10(train=True, raw=True, return_loader=False)

    # Get test set loader
    test_loader = load_cifar10(train=False)
    """

    def _make_loader(ds, shuffle=False):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    transform_list = [transforms.ToTensor()]

    if return_loader:
        # practical for smaller models
        meanR, meanG, meanB = 0.5, 0.5, 0.5
        stdR, stdG, stdB = 0.5, 0.5, 0.5
        # from data exploration - theoretically better
        # meanR, meanG, meanB = 0.4914, 0.4822, 0.4465
        # stdR, stdG, stdB = 0.2023, 0.1994, 0.2010
        transform_list.append(
            transforms.Normalize((meanR, meanG, meanB), (stdR, stdG, stdB))
        )
    if augmentation and train:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ]  # TODO: readme standard for CIFAR [https://arxiv.org/abs/1512.03385], https://github.com/DeepVoltaire/AutoAugment, https://huggingface.co/jaeunglee/resnet18-cifar10-unlearning
        )

    transform = transforms.Compose(transform_list)

    if raw:
        trainset = CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
        return _make_loader(trainset) if return_loader else trainset

    if train:
        trainset = CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

        if validation_split > 0:
            validation_size = int(len(trainset) * validation_split)
            train_size = len(trainset) - validation_size
            generator = torch.Generator().manual_seed(42)  # for reproducibility
            trainset, evalset = torch.utils.data.random_split(
                dataset=trainset,
                lengths=[train_size, validation_size],
                generator=generator,
            )
        else:
            evalset = CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )

        return (
            (_make_loader(trainset, shuffle=True), _make_loader(evalset))
            if return_loader
            else (trainset, evalset)
        )
    else:
        evalset = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        return _make_loader(evalset) if return_loader else evalset
