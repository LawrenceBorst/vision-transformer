import torch


class MultiHeadSelfAttentionBlock(torch.nn.Module):
    """
    Multi-head Self Attention mechanism.

    Args:
        embedding_dim (int): The embedding dimension. Defaults to 768.
        num_heads (int): The number of heads. Defaults to 12
        device (torch.device): The device to run on. Defaults to CPU.
    """

    _layer_norm: torch.nn.LayerNorm
    _multihead_attn: torch.nn.MultiheadAttention

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self._layer_norm = torch.nn.LayerNorm(
            normalized_shape=embedding_dim, device=device
        )
        self._multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0,
            batch_first=True,
            device=device,
        )
        self._device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer_norm(x)
        attn_output, _ = self._multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )

        return attn_output
