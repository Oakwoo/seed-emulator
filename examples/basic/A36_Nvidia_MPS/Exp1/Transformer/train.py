import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
import os
import pandas as pd

# ---- Configuration ---- #
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
SMOOTH_WINDOW = 50

# ---- Dataset ---- #
print("[INFO] Loading dataset...")
dataset = load_dataset("imdb")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(encoded_dataset['train'], batch_size=BATCH_SIZE, shuffle=True)

# ---- Model ---- #
print("[INFO] Initializing model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# ---- Training ---- #
losses = []
times = []
start_time = time.time()

print(f"[INFO] Training on device: {DEVICE}")
model.train()

for epoch in range(EPOCHS):
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
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
