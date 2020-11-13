from typing import Type
import torch
import numpy as np
from .utils import init_dataloaders


class trainer:

    def __init__(self,
                 model: Type[torch.nn.Module],
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 num_epochs: int = 200,
                 savename: str = 'model',
                 **kwargs):
        seed = kwargs.get("seed", 1)
        rng_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=kwargs.get("lr", 1e-3))
        self.criterion = torch.nn.MSELoss()
        (self.train_iterator,
         self.test_iterator) = init_dataloaders(
             X_train, y_train, X_test, y_test, kwargs.get("batch_size", 64))
        self.num_epochs = num_epochs
        self.filename = savename
        self.train_losses = []
        self.test_losses = []

    def train_step(self, feat: torch.Tensor,
                   tar: torch.Tensor) -> torch.Tensor:
        self.model.train()
        pred = self.model.forward(feat)
        loss = self.criterion(pred, tar)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def test_step(self, feat: torch.Tensor,
                  tar: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            pred = self.model.forward(feat)
            loss = self.criterion(pred, tar)
        return loss.item()

    def step(self) -> None:
        c, c_test = 0, 0
        losses, losses_test = 0, 0
        for feature, target in self.train_iterator:
            b = feature.size(0)
            losses += self.train_step(feature, target)
            c += b
        self.train_losses.append(losses / c)

        for feature, target in self.test_iterator:
            b = feature.size(0)
            losses_test += self.test_step(feature, target)
            c_test += b
        self.test_losses.append(losses_test / c_test)

    def run(self) -> Type[torch.nn.Module]:
        template = 'Epoch: {}... Training loss: {}... Test loss: {}'
        for e in range(self.num_epochs):
            self.step()
            print(template.format(
                e+1, np.round(self.train_losses[-1], 5),
                np.round(self.test_losses[-1], 5)))
        self.save_weights()
        return self.model.cpu()

    def save_weights(self, *args: str) -> None:
        try:
            filename = args[0]
        except IndexError:
            filename = self.filename
        torch.save(self.model.state_dict(),
                   filename + '.pt')


def rng_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # for GPU
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
