# Splits the data into training, validation and test sets.

import os


classes: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

TRAIN_DIR: str = "static/MNIST/train"
TEST_DIR: str = "static/MNIST/test"
VALID_DIR: str = "static/MNIST/valid"


def create_validation_set() -> None:
    """
    Split the data into training, validation and test sets

    The data, by default, comes with 10,000 test images and 60,000 training images.
    Let us arbitrarily go for a 10/10/80 split, meaning 7000/7000/56000 images
    """

    _create_validation_folder(VALID_DIR)

    _move_images(TRAIN_DIR, VALID_DIR, 3_000)
    _move_images(TEST_DIR, VALID_DIR, 4_000)

    return


def _create_validation_folder(dir: str) -> None:
    """
    Create the validation folder
    """

    os.makedirs(dir, exist_ok=True)

    for number in range(0, 10):
        os.makedirs(os.path.join(dir, classes[number]), exist_ok=True)

    return


def _move_images(source: str, dest: str, n: int) -> None:
    """
    Move images from source to dest folder
    And automatically number them starting from 1
    """

    for number in range(10):
        max_images: int = int(n / 10)  # Validation set will have
        # the same number of images for each class, though this is not really true
        # for the training or test sets. But it's not that uneven, so not a big deal
        class_name: str = classes[number]

        folder_size: int = len(os.listdir(os.path.join(source, class_name)))

        for idx in range(folder_size - max_images, folder_size):
            os.rename(
                os.path.join(source, class_name, f"{idx}.png"),
                os.path.join(dest, class_name, f"{idx - folder_size + max_images}.png"),
            )

    return


if __name__ == "__main__":
    create_validation_set()
