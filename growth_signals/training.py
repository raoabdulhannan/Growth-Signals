from tqdm import tqdm
from constants import HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, EMBEDDING_DIM, EPOCHS, LR
from plot import plot_dead_latents
import torch.optim as optim
from torch.utils.data import DataLoader
from sae import SAE, loss_function
import torch
from custom_dataset import CustomDataset
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(params):
    lr, batch_size, lambda_coef, hidden_dim, decoder_activation = params

    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    model = SAE(input_dim=768, hidden_dim=hidden_dim, lambda_coef=lambda_coef,
                decoder_activation=decoder_activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0
    for batch in tqdm(data_loader, desc="Training Batch"):
        batch = batch.to(device)
        optimizer.zero_grad()

        encoded, decoded = model(batch)
        loss = loss_function(decoded, batch, encoded, model)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss}")

    return avg_loss


def main():
    print("\n Training Final Model with Best Hyperparameters")

    dataset = CustomDataset()
    print("\n Custom Dataset Loaded")
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    print(f"\n Data processed {type(data_loader)}")

    model = SAE(
        input_dim=768,
        hidden_dim=HIDDEN_DIM,
        lambda_coef=0.1,
        decoder_activation="tanh"
    ).to(device)

    model = torch.compile(model)  # Optimize execution
    print("\n Model loaded")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("\n Optimizer loaded")

    dead_latents_per_epoch = []
    latent_activation_distribution = []

    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        total_loss = 0
        activations = []

        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch_embeddings, batch_texts = batch
            batch_embeddings = batch_embeddings.to(device)
            optimizer.zero_grad()

            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            activations.append(encoded.detach().cpu().numpy())
            print
            dead_latents = np.sum(activations == 0, axis=0) == len(activations)
            num_dead_latents = np.sum(dead_latents)

            dead_latents_per_epoch.append(num_dead_latents)
            activs = np.array(activations)
            latent_activation_distribution.append(np.sum(activs > 0, axis=1))

        plot_dead_latents(dead_latents_per_epoch=dead_latents_per_epoch)
        activations = np.concatenate(activations, axis=0)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(data_loader)}")


if __name__ == "__main__":
    main()
