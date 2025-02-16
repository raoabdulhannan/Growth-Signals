import numpy as np
import matplotlib.pyplot as plt


def plot_dead_latents(dead_latents_per_epoch):
    epochs = np.arange(1, len(dead_latents_per_epoch) + 1)

    plt.style.use("ggplot")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dead_latents_per_epoch, marker='o', linestyle='-', color='crimson', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgewidth=2, markeredgecolor='crimson', label="Dead Latents")
    
    plt.fill_between(epochs, dead_latents_per_epoch, alpha=0.2, color='crimson')

    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Dead Latents", fontsize=14, fontweight="bold")
    plt.title("Dead Latents Over Training Epochs", fontsize=16, fontweight="bold")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="upper right")
    
    plt.show()
