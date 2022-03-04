import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
learning_rate = 0.01
output_size = 10
batch_size = 100
num_epochs = 5


def get_transform():
    """ Gets the transforms to be applied to data set

    Returns a composition of transforms to be used in datasets
    """
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transforms


class ConvDigitRecognizer(nn.Module):
    """ A Cnn model for classifying mnist digits
    
    a) Uses 2 conv layers with kernel of 5x5 and ReLU
    b) Uses 2 linear layers with a ReLU
    """
    def __init__(self, input_channels, num_classes):
        super(ConvDigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.lin1 = nn.Linear(16*4*4, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


def run_eval(model, test_dl, epoch):
    with torch.no_grad():
        num_samples = 0
        num_correct = 0
        for i, (samples, labels) in enumerate(test_dl):
            images = samples.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            num_correct += (predictions == labels).sum().item()
            num_samples += labels.shape[0]

        acc = (100 * num_correct) / num_samples
        print(f'acc = {acc:.3f}, at epoch {epoch}')


def main():
    transforms = get_transform()

    # Prepare datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model = ConvDigitRecognizer(1, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    num_steps = len(train_dl)

    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_dl):
            x = samples.to(device)
            y = labels.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}, loss = {loss.item():.4f}')
    
        run_eval(model, test_dl, epoch)


if __name__ == '__main__':
    sys.exit(main())