def get_embedding_dim(in_channels: int, patch_size: int) -> int:
    embedding_dim: int = in_channels * patch_size**2

    return embedding_dim
