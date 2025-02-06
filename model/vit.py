import torch
from typing import List

from model.patch_embedding import PatchEmbedding
from model.transformer_encoder_block import TransformerEncoderBlock
from model.utils import get_embedding_dim


class ViT(torch.nn.Module):
    """
    A Vision Transformer (ViT) model.

    Args:
        img_size (List): The size of the image. Defaults to [224, 224].
        in_channels (int): The number of color channels. Defaults to 3.
        patch_size (int): The size of the patches. Defaults to 16.
        n_transformer_layers (int): The number of transformer layers. Defaults to 12.
        mlp_hidden_size (int): The size of the hidden layer in the MLP. Defaults to 3072.
        n_heads (int): The number of heads in the multihead attention. Defaults to 12.
        mlp_dropout (float): The dropout rate in the MLP. Defaults to 0.1.
        embedding_dropout (float): The dropout rate in the embedding. Defaults to 0.1.
        n_classes (int): The number of classes. Defaults to 10.
        device (torch.device): The device to run the model on. Defaults to torch.device
    """

    _patch_embedding: PatchEmbedding
    _embedding_dropout: torch.nn.Dropout
    _transformer_encoder: torch.nn

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        in_channels: int = 3,
        patch_size: int = 16,
        n_transformer_layers: int = 12,
        mlp_hidden_size: int = 3072,
        n_heads: int = 12,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        n_classes: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        embedding_dim: int = get_embedding_dim(in_channels, patch_size)

        self._patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            img_size=img_size,
            device=device,
        )
        self._embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self._transformer_encoder = torch.nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=n_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_dropout=mlp_dropout,
                    device=device,
                )
                for _ in range(n_transformer_layers)
            ]
        )
        self._classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=embedding_dim, device=device),
            torch.nn.Linear(
                in_features=embedding_dim, out_features=n_classes, device=device
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embedding_dropout(x)
        x = self._patch_embedding(x)
        x = self._embedding_dropout(x)
        x = self._transformer_encoder(x)
        x = self._classifier(x[:, 0])

        return x
