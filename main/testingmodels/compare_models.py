import os, json
import matplotlib.pyplot as plt
from pathlib import Path

model_names = [f"multiple_stocks_model_5min_{k}" for k in range(1, 9)]

root = Path.cwd()

models_path = root/'main'/'models'

with open(os.path.join(models_path, "model_info.json"), "r") as f:
    hist = json.load(f)

colors = plt.cm.get_cmap('tab20')

fig, ax = plt.subplots()

# Cycle through colors automatically
for idx, model_name in enumerate(model_names):
    json_entry = hist[model_name]
    
    training_errors = json_entry["training_loss_per_epoch"]
    validation_errors = json_entry["validation_loss_per_epoch"]
    
    epochs = range(1, len(training_errors) + 1)
    
    color = colors(idx % 10)  # Cycle color index inside colormap limits
    
    # Plot training loss
    ax.semilogy(epochs, training_errors, label=f"{model_name} Train", color=color, linestyle='-')
    
    # Plot validation loss
    ax.semilogy(epochs, validation_errors, label=f"{model_name} Val", color=color, linestyle='--')

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Squared Error (Loss)")
ax.set_title("Training and Validation Loss per Model")
ax.legend(loc='upper right')
ax.grid(True)

plt.show()
