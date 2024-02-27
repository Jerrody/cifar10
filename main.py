import os

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

dataset_path: str = './data/'
is_folder_empty: bool = not os.listdir(dataset_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Found device for fitting: {device}")


def imshow(img):
    np_img = img.numpy()
    np_img = (np_img - np.min(np_img)) / (np.max(np_img) - np.min(np_img))
    np_img = np.transpose(np_img, (1, 2, 0))
    plt.imshow(np_img)
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


class AdaptiveDropout(torch.nn.Module):
    current_epoch: int = 0

    def __init__(self, total_epochs, initial_drop_prob=1e-6, final_drop_prob=0.8, adaptivity_control_start=0.5,
                 adaptivity_control_end=0.1):
        super(AdaptiveDropout, self).__init__()
        self.initial_drop_prob = initial_drop_prob
        self.final_drop_prob = final_drop_prob
        self.adaptivity_control_start = adaptivity_control_start
        self.adaptivity_control_end = adaptivity_control_end
        self.total_epochs = total_epochs
        self.drop_prob = initial_drop_prob
        self.adaptivity_control = adaptivity_control_start

    def forward(self, x):
        self.update_parameters(self.current_epoch)

        if self.training:
            active_prob = x.abs().mean(dim=tuple(range(1, x.dim())), keepdim=True)
            dropout_prob = torch.clamp(self.drop_prob + self.adaptivity_control * (active_prob - self.drop_prob), 0, 1)
            mask = torch.bernoulli(1.0 - dropout_prob).to(x.device)
            x = x * mask / (1.0 - dropout_prob)
        return x

    def update_parameters(self, current_epoch):
        epoch_ratio = current_epoch / self.total_epochs
        self.drop_prob = (self.final_drop_prob - self.initial_drop_prob) * epoch_ratio + self.initial_drop_prob
        self.adaptivity_control = ((self.adaptivity_control_end - self.adaptivity_control_start) * epoch_ratio
                                   + self.adaptivity_control_start)

    def step(self):
        self.current_epoch += 1


class AdaptiveDropout2D(torch.nn.Module):
    def __init__(self, total_epochs, initial_drop_prob=1e-6, final_drop_prob=0.8, adaptivity_control_start=0.5,
                 adaptivity_control_end=0.1):
        super(AdaptiveDropout2D, self).__init__()
        self.initial_drop_prob = max(0.0, min(initial_drop_prob, 1.0))
        self.final_drop_prob = max(0.0, min(final_drop_prob, 1.0))
        self.adaptivity_control_start = adaptivity_control_start
        self.adaptivity_control_end = adaptivity_control_end
        self.total_epochs = total_epochs
        self.drop_prob = self.initial_drop_prob
        self.adaptivity_control = adaptivity_control_start
        self.current_epoch = 0

    def forward(self, x):
        self.update_parameters(self.current_epoch)

        if self.training:
            active_prob = x.abs().mean(dim=(2, 3), keepdim=True) if x.dim() > 3 else x.abs().mean(
                dim=tuple(range(1, x.dim())), keepdim=True)

            dropout_prob = torch.clamp(self.drop_prob + self.adaptivity_control * (active_prob - self.drop_prob), 0, 1)

            mask_shape = (x.shape[0], x.shape[1], 1, 1) if x.dim() > 3 else (x.shape[0], 1)
            mask = torch.bernoulli(1.0 - dropout_prob.expand_as(x)).to(x.device)

            x = x * mask / (1.0 - dropout_prob + 1e-6)

        return x

    def update_parameters(self, current_epoch):
        epoch_ratio = min(max(current_epoch / self.total_epochs, 0), 1)  # Clamp ratio to [0, 1]
        self.drop_prob = (self.final_drop_prob - self.initial_drop_prob) * epoch_ratio + self.initial_drop_prob
        self.adaptivity_control = ((self.adaptivity_control_end - self.adaptivity_control_start) * epoch_ratio
                                   + self.adaptivity_control_start)

    def step(self):
        self.current_epoch = min(self.current_epoch + 1, self.total_epochs)


class HomoeostaticRegulation(torch.nn.Module):
    def __init__(self, target_activity=0.1):
        super(HomoeostaticRegulation, self).__init__()
        self.target_activity = target_activity

    def forward(self, x):
        if self.training:
            current_activity = x.abs().mean()
            adjustment_factor = self.target_activity / current_activity
            x = x * adjustment_factor

        return x


class AggregationBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(AggregationBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.bn = torch.nn.BatchNorm2d(self.conv1.out_channels)
        self.dropout = torch.nn.Dropout2d(p=0.65)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        if x.size()[2:] != y.size()[2:]:
            y = torch.nn.functional.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)

        return self.dropout(x + y)


class Net(torch.nn.Module):
    def __init__(self, input_size=(3, 32, 32)):
        super().__init__()

        self.avg_pool = torch.nn.AvgPool2d(2, 1)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=2)
        self.bn1 = torch.nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = torch.nn.Conv2d(self.conv1.out_channels, 128, kernel_size=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = torch.nn.Conv2d(self.conv2.out_channels, 128, kernel_size=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = torch.nn.Conv2d(self.conv3.out_channels, 256, kernel_size=2, stride=2)
        self.bn4 = torch.nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = torch.nn.Conv2d(self.conv4.out_channels, 256, kernel_size=2, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(self.conv5.out_channels)

        self.dropout1 = torch.nn.Dropout2d(p=0.8)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.dropout3 = torch.nn.Dropout2d(p=0.1)

        self.adaptive_dropout_2d = AdaptiveDropout(num_epochs,
                                                   initial_drop_prob=0.2,
                                                   final_drop_prob=0.8,
                                                   adaptivity_control_start=0.2,
                                                   adaptivity_control_end=0.7)

        self.block_1 = AggregationBlock(in_channels=self.conv2.out_channels)
        self.block_2 = AggregationBlock(in_channels=self.conv4.out_channels)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = self.forward_conv(dummy_input)
            out_features = x.view(x.size(0), -1).size(1)

        self.fc1 = torch.nn.Linear(out_features, 128)
        self.bn5_1d = torch.nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, 10)

    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x1 = self.conv2(x)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)

        x2 = self.conv3(x1)
        x2 = self.bn3(x2)
        x2 = self.relu(x2)

        x = self.block_1.forward(x1, x2)
        x = self.avg_pool(x)
        # x = self.adaptive_dropout_2d(x)

        x3 = self.conv4(x)
        x3 = self.bn4(x3)
        x3 = self.relu(x3)

        x4 = self.conv5(x3)
        x4 = self.bn5(x4)
        x4 = self.relu(x4)

        x = self.block_2.forward(x3, x4)
        x = self.avg_pool(x)
        # x = self.dropout1(x)

        return x

    def forward(self, x):
        x = self.forward_conv(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn5_1d(x)
        x = self.relu(x)
        # x = self.dropout2(x)

        x = self.fc2(x)

        return x

    def step(self):
        self.adaptive_dropout_2d.step()
        # self.adaptive_dropout2.step()


train_batch_size = 128
test_batch_size = 64

temp_transform = transforms.Compose([transforms.ToTensor()])

_, temp_train_data_loader = get_loader(True, train_batch_size, shuffle=True, transform=temp_transform)
_, temp_test_data_loader = get_loader(False, test_batch_size, shuffle=False, transform=temp_transform)

train_mean, train_std = get_normalized_data_transform(temp_train_data_loader)
print(f"train_mean: {train_mean}, train_std: {train_std}")

test_mean, test_std = get_normalized_data_transform(temp_test_data_loader)
print(f"test_mean: {test_mean}, test_mean: {test_std}")

train_transform = transforms.Compose([
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std)
])

train_dataset, _ = get_loader(True, train_batch_size, shuffle=True, transform=train_transform)
dataset_len = len(train_dataset)
train_size = int(dataset_len * 0.9)
val_size = dataset_len - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

_, test_loader = get_loader(False, test_batch_size, shuffle=False, transform=test_transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(test_batch_size)))

lr = 1e-3
weight_decay = 1e-1
num_epochs = 100

net = Net()
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, verbose=True)

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

    scheduler.step(val_loss)
    net.step()

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

dataiter = iter(test_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
