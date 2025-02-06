import click
from engine.engine import Engine
import setup
import torch
from torchvision import transforms
from data import create_data_loaders
from model.vit import ViT


@click.command()
@click.option("--debug", is_flag=True, help="Run in debug mode")
def main(debug: bool) -> None:
    DEBUG: bool = debug
    if DEBUG:
        print("Running in debug mode")

    setup.set_seed()
    device: torch.device = setup.get_device()
    CPU_COUNT: int = setup.get_cpu_count() if not DEBUG else 1
    MNIST_STATS: dict = {"std": [0.3081], "mean": [0.1307]}
    BATCH_SIZE: int = 32 if not DEBUG else 2

    train_dir: str = "static/MNIST/train"
    test_dir: str = "static/MNIST/test"
    height: int = 224
    width: int = 224
    color_channels: int = 3
    patch_size: int = 16

    train_loader, test_loader, classes = create_data_loaders(
        train_dir=train_dir,
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
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        model=vit_model,
        device=device,
    )

    results = engine.train()

    print(results)

    return


if __name__ == "__main__":
    main()
