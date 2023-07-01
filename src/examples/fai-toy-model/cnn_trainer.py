import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from cnn_model import CNN
import pytorch_lightning as pl


class CNNLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.cnn = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.classes = ['plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        predictions = self.cnn(inputs)
        return self.criterion(predictions, labels)

    def validation_step(self, batch, batch_idx):
        return batch

    def validation_epoch_end(self, batches) -> None:
        batch = batches[0]
        inputs, labels = batch
        predictions = self.cnn(inputs)
        loss = self.criterion(predictions, labels)
        print("Val Loss:", loss.item())
        predictions = torch.argmax(predictions, dim=-1)
        for i in range(20):
            print("Prediction", predictions[i].item())
            print("Ground Truth", labels[i].item())
            print()

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    cnn = CNNLightning()
    trainer = pl.Trainer(max_steps=10000, gpus=1)
    trainer.fit(cnn, trainloader, valloader)
