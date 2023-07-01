import torch
from torch import nn
from torch import optim
from torch._C import Size
import torchvision
import torchvision.transforms as transforms
from cvae_model import CVAE
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.utils import save_image


class CVAELightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.cvae = CVAE(1, 10, 100)
        # self.classes = ['plane', 'car', 'bird', 'cat',
        #                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def onehot(self, labels, num_classes=10):
        targets = torch.zeros(labels.size(0), num_classes)
        for i, label in enumerate(labels):
            targets[i, label] = 1
        return targets.to(self.args["device"])

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.cvae(inputs, labels=self.onehot(labels))
        loss = self.cvae.loss_function(*results, M_N=inputs.shape[0] / self.args["len_train_dataset"])
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.cvae(inputs, labels=self.onehot(labels))
        loss = self.cvae.loss_function(*results, M_N=inputs.shape[0] / self.args["len_train_dataset"])
        return batch

    def validation_epoch_end(self, batches) -> None:
        batch = batches[0]
        inputs, labels = batch
        results = self.cvae(inputs, labels=self.onehot(labels))
        save_image(results[0][0].view(1, 3, 32, 32), 'samples/sample_' + str(labels[0].item()) + '.png')

    def configure_optimizers(self):
        return optim.Adam(self.cvae.parameters(),
                          lr=3e-5)


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5)),
         ])

    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    args = {"batch_size": 32, "device": device, "len_train_dataset": len(train_loader)}

    cnn = CVAELightning(args)
    trainer = pl.Trainer(max_steps=100000, gpus=1)
    trainer.fit(cnn, train_loader, test_loader)
