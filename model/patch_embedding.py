from typing import List
import torch

from model.utils import get_embedding_dim


class PatchEmbedding(torch.nn.Module):
    """This class converts an image into a sequence of learnable patch embeddings

    Args:
        in_channels (int): Number of color channels. Defaults to 3.
        patch_size (int): Size of patches. Defaults to 16.
        img_size (List): Size of the image. Defaults to [224, 224].
    """

    _embedding_dim: int
    _patch_size: int
    _img_size: List[int]
    _patcher: torch.nn.Conv2d
    _flatten: torch.nn.Flatten
    _device: torch.device

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        img_size: List[int] = [224, 224],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        [img_h, img_w] = img_size
        assert (
            img_w % patch_size == 0 and img_h % patch_size == 0
        ), f"""Input image size must be divisible by patch size,
        image shape: {(img_h, img_w)}, patch size: {patch_size}"""

        super().__init__()

        self._embedding_dim = get_embedding_dim(in_channels, patch_size)
        self._device = device

        # Treat the series as an image of patches that are mapped to the transformer
        # input via a learnable embedding. This is achievable with a convolution
        # followed by a suitable linear mapping
        self._patcher = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            device=self._device,
        )
        self._img_size = img_size
        self._patch_size = patch_size

        # Flatten the spatial dimensions
        self._flatten = torch.nn.Flatten(
            start_dim=2,
            end_dim=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch embedding layer.
        Expects (image) input of shape
        (batch_size, in_channels, image_resolution, image_resolution).
        """
        raw_embeddings: torch.Tensor = self._get_raw_patch_embeddings(x)
        class_token: torch.Tensor = self._get_class_token(x)
        position_embeddings: torch.Tensor = self._get_position_embeddings(
            self._img_size[0], self._img_size[1]
        )

        embedding_with_class_token: torch.Tensor = torch.cat(
            [class_token, raw_embeddings], dim=1
        )

        return embedding_with_class_token + position_embeddings

    def _get_raw_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x_patched: torch.Tensor = self._patcher(x)
        return self._flatten(x_patched).permute(0, 2, 1)

    def _get_class_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the class token for the image
        """
        batch_size: int = x.shape[0]
        embedding_length: int = self._embedding_dim

        return torch.nn.Parameter(
            torch.randn(batch_size, 1, embedding_length),
            requires_grad=True,
        ).to(self._device)

    def _get_position_embeddings(self, img_width: int, img_height: int) -> torch.Tensor:
        n_patches: int = int(img_width * img_height / self._patch_size**2)

        return torch.nn.Parameter(
            torch.randn(1, n_patches + 1, self._embedding_dim), requires_grad=True
        ).to(self._device)
