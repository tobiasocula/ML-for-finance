import os, json, keras, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model_names = [f"DaySin_OHCL_model_{k}" for k in range(1, 9)]

script_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(os.path.dirname(script_dir), 'models')
colors = plt.cm.get_cmap('tab20')

with open(os.path.join(models_path, "hist.json"), "r") as f:
    hist = json.load(f)

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
