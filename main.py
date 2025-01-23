import setup
import torch
from torchvision import transforms
from data import create_data_loaders


def main() -> None:
    setup.set_seed()
    device: torch.device = setup.get_device()
    CPU_COUNT: int = setup.get_cpu_count()
    MNIST_STATS: dict = {"std": [0.3081], "mean": [0.1307]}

    train_loader, test_loader, classes = create_data_loaders(
        train_dir="static/MNIST/train",
        test_dir="static/MNIST/test",
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(**MNIST_STATS)]
        ),
        num_workers=CPU_COUNT,
    )

    return


if __name__ == "__main__":
    main()
