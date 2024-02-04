﻿import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

dataset_path: str = './data/'
is_folder_empty: bool = not os.listdir(dataset_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Found device for fitting: {device}")


def imshow(img):
    np_img = img.numpy()
    np_img = np_img / np.max(np_img)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def get_normalized_data_transform(dataset: torch.utils.data.DataLoader):
    channels_sum: float = 0
    channels_squared_sum: float = 0
    num_batches: float = 0

    for data, _ in dataset:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean: float = channels_sum / num_batches
    std: float = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_loader(is_train: bool, batch_size: int, shuffle: bool, transform: transforms.Compose):
    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=is_train, download=is_folder_empty,
                                           transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=0)

    return dataset, loader


class Net(torch.nn.Module):
    bn5_2d: torch.nn.BatchNorm2d

    bn6_1d: torch.nn.BatchNorm1d
    bn7_1d: torch.nn.BatchNorm1d

    def __init__(self):
        super().__init__()

        self.pool = torch.nn.MaxPool2d(2, 1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = torch.nn.Conv2d(self.conv1.out_channels, 12, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = torch.nn.Conv2d(self.conv2.out_channels, 24, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = torch.nn.Conv2d(self.conv3.out_channels, 36, kernel_size=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = torch.nn.Conv2d(self.conv4.out_channels, 48, kernel_size=2, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(self.conv5.out_channels)

        # TODO: Calculate in features.
        self.fc1 = torch.nn.Linear(49152, 1024)
        self.bn6_1d = torch.nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, 2048)
        self.bn7_1d = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn6_1d(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn7_1d(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


train_batch_size = 10
test_batch_size = 4

temp_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

_, temp_train_data_loader = get_loader(True, train_batch_size, shuffle=True, transform=temp_transform)
_, temp_test_data_loader = get_loader(False, test_batch_size, shuffle=False, transform=temp_transform)

train_mean, train_std = get_normalized_data_transform(temp_train_data_loader)
print(f"train_mean: {train_mean}, train_std: {train_std}")

test_mean, test_std = get_normalized_data_transform(temp_test_data_loader)
print(f"test_mean: {test_mean}, test_mean: {test_std}")

transform_ops = [transforms.RandomVerticalFlip(),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(),
                 transforms.RandomAutocontrast(),
                 transforms.RandomPerspective()]

train_transform = transforms.Compose(
    transform_ops + [transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(test_mean, test_std)])

train_dataset, _ = get_loader(True, train_batch_size, shuffle=True, transform=train_transform)
dataset_len = len(train_dataset)
train_size = int(dataset_len * 0.8)
val_size = dataset_len - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

_, test_loader = get_loader(False, test_batch_size, shuffle=False, transform=test_transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(test_batch_size)))

lr = 3e-4
gamma = 0.7
num_epochs = 50

net = Net()
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

print('LEARNING')
writer = SummaryWriter()

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    net.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/Epoch', loss, epoch)
    writer.add_scalar('Train/Epoch', train_accuracy, epoch)
    writer.add_scalar('Validation/Epoch', val_accuracy, epoch)

    print(f'Epoch {epoch + 1}/{num_epochs} - '
          f'Train Loss: {loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {val_accuracy:.2f}%')
writer.flush()

print('Finished Training')
print('Testing')

net.eval()

total = 0
correct = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test accuracy: {test_accuracy:.2f}%')
