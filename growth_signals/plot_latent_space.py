import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import kendalltau
from scipy.spatial.distance import cosine


def plot_re_ranked_vectors(encoded_vectors, titles, epoch, step,
                           save_dir="latent_plots"):
    os.makedirs(save_dir, exist_ok=True)

    encoded_vectors = encoded_vectors.detach().cpu().numpy()

    first_vector = encoded_vectors[0]
    ranking_order = np.argsort(-first_vector)
    re_ranked_vectors = encoded_vectors[:, ranking_order]

    plt.figure(figsize=(10, 6))
    for i, vector in enumerate(re_ranked_vectors):
        plt.plot(vector, label=titles[i])

    plt.title(f"Re-ranked Latent Space Vectors (Epoch {epoch}, Step {step})")
    plt.suptitle(f"Title for First Vector: {titles[0]}", fontsize=10, y=0.95)
    plt.xlabel("Re-ranked Latent Space")
    plt.ylabel("Value")
    plt.legend(loc="center left", bbox_to_anchor=(0.9, 0.5),
               bbox_transform=plt.gcf().transFigure,
               fontsize="small", ncol=1)
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(save_dir,
                                 f"re_ranked_vectors_epoch_{epoch}_step_{step}.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()

    plot_faceted_re_ranked_vectors(re_ranked_vectors, titles, epoch, step, save_dir)
    plot_rank_correlation_heatmap(re_ranked_vectors, titles, epoch, step, save_dir)


def plot_faceted_re_ranked_vectors(re_ranked_vectors, titles, epoch, step, save_dir):
    num_vectors = len(re_ranked_vectors)
    cols = 2
    rows = (num_vectors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, vector in enumerate(re_ranked_vectors):
        ax = axes[i]
        ax.plot(vector)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlabel("Latent Dimension (Re-ranked)")
        ax.set_ylabel("Value")
        ax.grid(True)

    for j in range(len(re_ranked_vectors), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Re-ranked Latent Space Vectors (Epoch {epoch}, Step {step})", fontsize=16)
    plot_filename = os.path.join(save_dir,
                                 f"faceted_re_ranked_vectors_epoch_{epoch}_step_{step}.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()


def plot_rank_correlation_heatmap(re_ranked_vectors, titles, epoch, step, save_dir="latent_plots"):
    num_vectors = len(re_ranked_vectors)
    rank_correlation_matrix = np.zeros((num_vectors, num_vectors))

    # Kendall's rank correlation
    for i in range(num_vectors):
        for j in range(num_vectors):
            corr, _ = kendalltau(re_ranked_vectors[i], re_ranked_vectors[j])
            rank_correlation_matrix[i, j] = corr

    # Cosine similarity
    cosine_similarity_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            cosine_similarity_matrix[i, j] = 1 - cosine(re_ranked_vectors[i], re_ranked_vectors[j])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    sns.heatmap(rank_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=titles, yticklabels=titles, cbar_kws={'label': "Kendall's Tau"}, ax=axes[0])
    axes[0].set_title(f"Kendall's Tau Heatmap (Epoch {epoch}, Step {step})", fontsize=14)

    sns.heatmap(cosine_similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=titles, yticklabels=titles, cbar_kws={'label': "Cosine Similarity"}, ax=axes[1])
    axes[1].set_title(f"Cosine Similarity Heatmap (Epoch {epoch}, Step {step})", fontsize=14)

    combined_filename = os.path.join(save_dir,
                                      f"combined_heatmaps_epoch_{epoch}_step_{step}.png")
    plt.savefig(combined_filename, bbox_inches="tight")
    plt.close()

