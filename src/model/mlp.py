import torch


class MLPBlock(torch.nn.Module):
    """
    An MLP block as described in the Vision Transformer paper.

    Args:
        embedding_dim (int): The embedding dimension. Defaults to 768.
        hidden_size (int): The hidden size. Defaults to 3072.
        dropout (float): The dropout rate. Defaults to 0.1.
        device (torch.device): The device to run on. Defaults to CPU.
    """

    _layer_norm: torch.nn.LayerNorm
    _mlp: torch.nn

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_size: int = 3072,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self._layer_norm = torch.nn.LayerNorm(
            normalized_shape=embedding_dim, device=device
        )
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embedding_dim, out_features=hidden_size, device=device
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(
                in_features=hidden_size, out_features=embedding_dim, device=device
            ),
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer_norm(x)

        return self._mlp(x)
