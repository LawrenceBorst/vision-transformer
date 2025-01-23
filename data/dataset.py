from typing import Annotated, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torch.types import Tensor
from torchvision.transforms import Compose
import pathlib
import os
from functools import cache


class MNIST(Dataset):
    """
    A custom dataset class for the MNIST dataset
    """

    _transform: None | Compose
    _target_dir: str

    def __init__(self, target_dir: str, transform: Optional[Compose] = None) -> None:
        self._transform = transform
        self._target_dir = target_dir

        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Directory '{target_dir}' does not exist")

        self._paths: list[pathlib.PosixPath] = list(
            pathlib.Path(target_dir).glob("*/*.png")
        )

        self._classes, self._classes_to_idx = self._find_classes(target_dir)

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

        return Image.open(img_path)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset
        """
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor | Image.Image, int]:
        """
        Returns the image and its class index

        If a transform is provided, the image is transformed
        """
        img: Image.Image = self._load_image(idx)
        cls_name: str = self._paths[idx].parent.name
        class_idx: int = self._classes_to_idx[cls_name]

        if self._transform:
            img = self._transform(img)

        return img, class_idx

    def _find_classes(
        self, target_dir: str
    ) -> Tuple[
        Annotated[list[str], "class names"], Annotated[dict[str, int], "class to index"]
    ]:
        """
        Finds the class folders in the target directory

        Args:
            target_dir (str): The target directory

        Returns:
            Tuple[list[str], dict[str, int]]: The classes and their indices

        Example:
            >>> MNIST._find_classes("static/MNIST/train")
            (['0', '1'], {'0': 0, '1': 1})
        """
        cls: list[str] = [d.name for d in os.scandir(target_dir) if d.is_dir()]

        cls_idx: dict[str, int] = {cls: idx for idx, cls in enumerate(cls)}

        return cls, cls_idx
