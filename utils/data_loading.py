# load cifar 10 dataset from torch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


def load_cifar10(
    train: bool,
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
