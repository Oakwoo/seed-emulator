import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# ---- Configuration ---- #
BATCH_SIZE = 128
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
SMOOTH_WINDOW = 10

# ---- Dataset ---- #
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model ---- #
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# ---- Training ---- #
losses = []
times = []
start_time = time.time()

print(f"[INFO] Training on device: {DEVICE}")

for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        times.append(time.time() - start_time)

    print(f"Epoch {epoch+1}/{EPOCHS} complete. Loss: {loss.item():.4f}")

# ---- Save Data ---- #
results = pd.DataFrame({"time": times, "loss": losses})
results.to_csv(f"results_{EXPERIMENT_NAME}.csv", index=False)
print(f"[INFO] Results saved to results_{EXPERIMENT_NAME}.csv")

# Optional: Smooth Plot
smoothed_losses = pd.Series(losses).rolling(window=SMOOTH_WINDOW).mean()
plt.plot(times, smoothed_losses, label=EXPERIMENT_NAME)
plt.xlabel("Time (s)")
plt.ylabel("Loss (Smoothed)")
plt.title("Training Loss vs Time")
plt.legend()
plt.grid(True)
plt.savefig(f"loss_curve_{EXPERIMENT_NAME}.png")
print(f"[INFO] Plot saved to loss_curve_{EXPERIMENT_NAME}.png")
