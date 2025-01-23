from torch.utils.data import DataLoader
from typing import Tuple, Optional
from data.dataset import MNIST
from torchvision import transforms


def create_data_loaders(
    train_dir: str,
    test_dir: str,
    transform: Optional[transforms.Compose] = None,
    num_workers: Optional[int] = 1,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """
    Create data loaders for the MNIST dataset.

    Args:
        train_dir (str): The directory to the training data.
        test_dir (str): The directory to the test data.
        transform (Optional[transforms.Compose]): The transformations to apply to the data.
        cpu_count (Optional[int]): The number of CPU cores to use.

    Returns:
        Tuple[DataLoader, DataLoader, list[str]]: A tuple containing the training data loader, the test data loader, and the classes.
    """
    train: MNIST = MNIST(
        target_dir=train_dir,
        transform=transform,
    )

    test: MNIST = MNIST(target_dir=test_dir, transform=transform)

    train_loader: DataLoader = DataLoader(
        train, batch_size=64, shuffle=True, num_workers=num_workers
    )
    test_loader: DataLoader = DataLoader(
        test, batch_size=64, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, train.classes
