from PIL import Image
from typing import Tuple
from torch import Tensor


ImageItem = Tuple[Tensor | Image.Image, int]
