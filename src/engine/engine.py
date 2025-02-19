from typing import Tuple
import torch
from tqdm.auto import tqdm


class Engine:
    """
    The engine class for training and evaluating a model

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        loss_fn (torch.nn.Module): The loss function to use.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device to train on
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device=torch.device,
    ):
        self._model = model
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._epochs = epochs
        self._device = device

    def train(
        self,
    ) -> dict:
        results: dict = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        for _ in tqdm(range(self._epochs)):
            train_loss, train_acc = self._train_step()
            test_loss, test_acc = self._test_step()

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        return results

    def _train_step(self) -> Tuple[float, float]:
        self._model.train()

        train_loss, train_acc = 0, 0

        for _, (X, y) in tqdm(enumerate(self._train_loader)):
            X, y = X.to(self._device), y.to(self._device)

            y_pred = self._model(X)

            loss = self._loss_fn(y_pred, y)
            train_loss += loss.item()

            self._optimizer.zero_grad()

            loss.backward()

            self._optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss = train_loss / len(self._train_loader)
        train_acc = train_acc / len(self._train_loader)

        return train_loss, train_acc

    def _test_step(self) -> Tuple[float, float]:
        self._model.eval()

        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for _, (X, y) in tqdm(enumerate(self._test_loader)):
                X, y = X.to(self._device), y.to(self._device)

                test_pred_logits = self._model(X)

                loss = self._loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss = test_loss / len(self._test_loader)
        test_acc = test_acc / len(self._test_loader)

        return test_loss, test_acc
