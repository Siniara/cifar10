# load cifar 10 dataset from torch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


def load_cifar10(
    train: bool,
    validation_split: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 2,
    return_loader: bool = True,
) -> DataLoader | Dataset | tuple[DataLoader, DataLoader] | tuple[Dataset, Dataset]:
    """
    Load the CIFAR-10 dataset. Can return either DataLoader objects or raw datasets for data exploration.

    Returns:
    - If train=True: (trainset, evalset) → trainset for training, evalset for validation or test
    - If train=False: evalset only → the official test set
    - If return_loader=True, returns DataLoader objects instead of datasets.
    """

    def _make_loader(ds, shuffle=False):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    transform_list = [transforms.ToTensor()]
    if return_loader:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    if not train:
        evalset = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        return _make_loader(evalset) if return_loader else evalset

    else:
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
