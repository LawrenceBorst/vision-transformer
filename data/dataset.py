from typing import Annotated, Optional, Tuple
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pathlib
import os
from functools import cache
from data.types import ImageItem


class MNIST(Dataset[ImageItem]):
    """
    A custom dataset class for the MNIST dataset

    Args:
        target_dir (str): The directory containing the dataset
        transform (Optional[Compose]): The transformations to apply to the data
        frac (float): Fraction of the dataset to use
    """

    _transform: None | Compose
    _target_dir: str
    _frac: float

    def __init__(
        self,
        target_dir: str,
        transform: Optional[Compose] = None,
        frac: float = 1.0,
    ) -> None:
        self._transform = transform
        self._target_dir = target_dir
        self._frac = frac

        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Directory '{target_dir}' does not exist")

        self._paths: list[pathlib.PosixPath] = list(
            pathlib.Path(target_dir).glob("*/*.png")
        )

        self._classes, self._classes_to_idx = self._find_classes()

    @property
    def classes(self) -> list[str]:
        return self._classes

    @property
    def classes_to_idx(self) -> dict[str, int]:
        return self._classes_to_idx

    @property
    @cache
    def idx_to_classes(self) -> dict[int, str]:
        return {v: k for k, v in self._classes_to_idx.items()}

    def _load_image(self, idx: int) -> Image.Image:
        img_path: pathlib.PosixPath = self._paths[idx]

        img: ImageFile = Image.open(img_path, mode="r", formats=["PNG"])
        img = img.convert("RGB")

        return img

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset
        """
        return int(len(self._paths) * self._frac)

    def __getitem__(self, idx: int) -> ImageItem:
        """
        Returns the image and its class index

        If a transform is provided, the image is transformed
        """
        img: Image.Image = self._load_image(idx)
        cls_name: str = self._paths[idx].parent.name
        class_idx: int = self._classes_to_idx[cls_name]

        if self._transform:
            # Remove last channel corresponding to alpha
            img = self._transform(img)

        return img, class_idx

    def _find_classes(
        self,
    ) -> Tuple[
        Annotated[list[str], "class names"], Annotated[dict[str, int], "class to index"]
    ]:
        """
        Obtains the class names and indices for this dataset

        Returns:
            Tuple[list[str], dict[str, int]]: The classes and their indices

        Example:
            >>> MNIST._find_classes("static/MNIST/train")
            (['0', '1'], {'0': 0, '1': 1})
        """
        cls: list[str] = [d.name for d in os.scandir(self._target_dir) if d.is_dir()]

        cls_idx: dict[str, int] = {cls: idx for idx, cls in enumerate(cls)}

        return cls, cls_idx
