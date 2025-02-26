import sys
from typing import Tuple
import click
from src.engine.engine import Engine
import src.setup as setup
import torch
from torchvision import transforms
from src.data import create_data_loaders
from src.model.vit import ViT


@click.command()
@click.option(
    "--debug",
    is_flag=True,
    help="Run in debug mode",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs to train on",
)
@click.option(
    "--limit_data",
    type=int,
    help="Limit number of samples to use in the train and test datasets",
)
def main(debug: bool, epochs: int, limit_data: int | None) -> None:
    epochs, limit_data = _use_constraints(epochs, limit_data)

    if debug:
        print("Running in debug mode")

    setup.set_seed()
    device: torch.device = setup.get_device()
    CPU_COUNT: int = 3
    MNIST_STATS: dict = {"std": [0.3081], "mean": [0.1307]}
    BATCH_SIZE: int = 32

    train_dir: str = "static/MNIST/train"
    valid_dir: str = "static/MNIST/valid"
    test_dir: str = "static/MNIST/test"
    height: int = 224
    width: int = 224
    color_channels: int = 3
    patch_size: int = 16

    train_loader, _, valid_loader, classes = create_data_loaders(
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((width, height)),
                transforms.ToTensor(),
                transforms.Normalize(**MNIST_STATS),
            ]
        ),
        num_workers=CPU_COUNT,
        batch_size=BATCH_SIZE,
        limit_data=limit_data,
    )

    vit_model = ViT(
        img_size=[height, width],
        in_channels=color_channels,
        patch_size=patch_size,
        n_transformer_layers=12,
        mlp_hidden_size=3072,
        n_heads=12,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
        n_classes=10,
        device=device,
    )

    optimizer = torch.optim.Adam(
        params=vit_model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    engine: Engine = Engine(
        train_loader=train_loader,
        test_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        model=vit_model,
        device=device,
    )

    engine.train()

    engine.save_model()

    return


def _use_constraints(epochs: int, limit_data: int | None) -> Tuple[int, int]:
    if epochs <= 0:
        raise Exception("Number of epochs must be greater than 0")

    if limit_data is None:
        limit_data = sys.maxsize

    if limit_data < 0:
        raise Exception("Dataset fraction must be less than or equal to 1")

    return epochs, limit_data


if __name__ == "__main__":
    main()
