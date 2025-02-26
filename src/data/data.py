from torch.utils.data import DataLoader
from typing import Tuple, Optional
from .dataset import MNIST
from torchvision import transforms
from .types import ImageItem


def create_data_loaders(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    transform: Optional[transforms.Compose] = None,
    num_workers: Optional[int] = 1,
    batch_size: Optional[int] = 64,
    limit_data: int | None = None,
) -> Tuple[DataLoader[ImageItem], DataLoader[ImageItem], list[str]]:
    """
    Create data loaders for the MNIST dataset.

    Args:
        train_dir (str): The directory to the training data.
        test_dir (str): The directory to the test data.
        transform (Optional[transforms.Compose]): The transformations to apply to
            the data.
        cpu_count (Optional[int]): The number of CPU cores to use.
        batch_size (Optional[int]): The batch size to use.
        limit_data (int | None): Maximum number of data samples to use

    Returns:
        Tuple[DataLoader, DataLoader, list[str]]: A tuple containing the training
            data loader, the test data loader, and the classes.
    """
    train: MNIST = MNIST(
        target_dir=train_dir,
        transform=transform,
        limit_data=limit_data,
    )
    valid: MNIST = MNIST(
        target_dir=valid_dir,
        transform=transform,
        limit_data=limit_data,
    )
    test: MNIST = MNIST(
        target_dir=test_dir,
        transform=transform,
        limit_data=limit_data,
    )

    train_loader: DataLoader[ImageItem] = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader: DataLoader[ImageItem] = DataLoader(
        valid, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader: DataLoader[ImageItem] = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader, train.classes
