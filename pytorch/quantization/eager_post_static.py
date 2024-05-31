import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torch.optim as optim



print(torch.__version__)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(((32//2) * (32//2) * 32), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.linear_relu_stack(x)
        logits = self.softmax(x)

        return logits



model = SimpleCNN()

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

learning_rate = 0.001
batch_size = 32
loss_fn = nn.CrossEntropyLoss()
epochs = 10
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size)

model(train_data[1][0].reshape((1,3,32,32)))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# test data
def test_loss(model, test_data):
    loss_sum = 0.0
    correct = 0
    total = 0
    test_data_loader = DataLoader(dataset=test_data, batch_size=32)
    for (x, y) in test_data_loader:
        y_pred = model(x)
        
        loss_sum += loss_fn(y_pred, y)
        total += y.size(0)
        correct += sum(y_pred.argmax(dim=1) == y)
    print("--= TESTING =--")
    print("Loss:", loss_sum.item() / len(test_data_loader))
    print("Accuracy:", correct.item() / (len(test_data_loader) * batch_size))

test_loss(model, test_data)

for ep in range(epochs):
    for x, y in train_data_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        test_loss(model, test_data)
