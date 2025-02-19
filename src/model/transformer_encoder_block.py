import torch

from .multihead_self_attention_block import MultiHeadSelfAttentionBlock
from .mlp import MLPBlock


class TransformerEncoderBlock(torch.nn.Module):
    """
    Transformer Encoder Block.

    This block combines a multihead self attention block with an MLP
    Introducing residual connections to aid training in later layers.

    Not much different from the in-built transformer encoder layer (had we used the
    same parameters) but good practice nonetheless.

    Args:
        embedding_dim (int): The embedding dimension. Defaults to 768.
        num_heads (int): The number of heads. Defaults to 12.
        mlp_hidden_size (int): The size of the hidden layer in the MLP. Defaults to
            3072.
        mlp_dropout (float): The dropout rate in the MLP. Defaults to 0.1.
        device (torch.device): The device to run on (defaults to CPU).
    """

    _msa_block: MultiHeadSelfAttentionBlock
    _mlp_block: MLPBlock

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_hidden_size: int = 3072,
        mlp_dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self._msa_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, device=device
        )
        self._mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            hidden_size=mlp_hidden_size,
            dropout=mlp_dropout,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._msa_block.forward(x) + x

        return self._mlp_block.forward(x) + x
