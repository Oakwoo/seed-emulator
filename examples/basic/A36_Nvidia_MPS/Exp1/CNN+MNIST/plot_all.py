import pandas as pd
import matplotlib.pyplot as plt
import glob

SMOOTH_WINDOW = 10

# Gather all result CSV files
files = glob.glob("results_*.csv")

plt.figure(figsize=(10, 6))

for file in files:
    df = pd.read_csv(file)
    label = file.split("results_")[1].split(".csv")[0]
    smoothed_loss = df["loss"].rolling(window=SMOOTH_WINDOW).mean()
    plt.plot(df["time"], smoothed_loss, label=label)

plt.xlabel("Time (s)")
plt.ylabel("Loss (Smoothed)")
plt.title("Training Loss vs Time (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_comparison_all.png")
print("[INFO] Comparison plot saved to loss_comparison_all.png")
