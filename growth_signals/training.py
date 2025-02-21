from tqdm import tqdm
from constants import HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, EMBEDDING_DIM, EPOCHS, LR
from plot import plot_dead_latents
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sae import SAE, loss_function
import torch
from custom_dataset import CustomDataset
# from skopt import gp_minimize
# from skopt.space import Integer, Real, Categorical
import numpy as np
import os
# import shutil
# import subprocess
import time
# from torch.utils.tensorboard import SummaryWriter
import visdom
import requests

"""
For running Visdom:
1. Install: pip install visdom
2. Run: python -m visdom.server in a separate terminal
3. Run the current script
(vis = visdom.Visdom(port=8097) will connect to the server you opened in step 2)
(All required code are already uncommented)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_checkpoint(epoch, model, optimizer, dataset_size, save_dir='./models'):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'{epoch}_{dataset_size}.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def wait_for_visdom():
    visdom_url = "http://localhost:8097"
    while True:
        try:
            requests.get(visdom_url)
            print("Visdom server is up and running.")
            break
        except requests.ConnectionError:
            print("Waiting for Visdom server...")
            time.sleep(1)


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

    # # Start Visdom server
    # visdom_process = subprocess.Popen(
    #     ["python", "-m", "visdom.server"],
    #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    # )
    # subprocess.Popen(["xdg-open", "http://localhost:8097"],
    #                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait_for_visdom()
    vis = visdom.Visdom(port=8097)
    vis_recon_loss = vis.line(Y=np.array([0]), X=np.array([0]),
                              opts=dict(title='Reconstruction Loss',
                                        xlabel='Step', ylabel='Loss',
                                        width=800, height=400))
    vis_dead_latents = vis.line(Y=np.array([0]), X=np.array([0]),
                                opts=dict(title='Dead Latents',
                                          xlabel='Epoch',
                                          ylabel='Number of Dead Latents'))
    # time.sleep(10)

    dataset = CustomDataset()

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    print("\n Custom Dataset Loaded")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)

    print(f"\n Data processed {type(train_loader)}")

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
    step = 0
    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        total_loss = 0
        activations = []
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch_embeddings = batch.to(device)
            optimizer.zero_grad()

            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Set loss values greater than 1 to 1
            loss_value = min(loss.item(), 1)
            epoch_losses.append(loss_value)

            if encoded.detach().cpu().numpy().shape[0] != BATCH_SIZE:
                print(f"Warning: Batch {step} has shape {encoded.detach().cpu().numpy().shape}, expected ({BATCH_SIZE}, latent_dim)")
    
            activations.append(encoded.detach().cpu().numpy())
            
            dead_latents = np.sum(activations == 0, axis=0) == len(activations)
            num_dead_latents = np.sum(dead_latents)

            dead_latents_per_epoch.append(num_dead_latents)
            activs = np.concatenate(activations, axis=0)
            latent_activation_distribution.append(np.sum(activs > 0, axis=1))
            # Log loss per batch
            # writer.add_scalar("Reconstruction Loss", loss.item(), step)
            # writer.flush()

            # Update Visdom plot for reconstruction loss (most recent 1000 data points)
            if step > 1000:
                vis.line(Y=np.array(epoch_losses[-1000:]), 
                         X=np.array(range(step-999, step+1)), 
                         win=vis_recon_loss, update='replace')
            else:
                vis.line(Y=np.array(epoch_losses), 
                         X=np.array(range(step-len(epoch_losses)+1, step+1)), 
                         win=vis_recon_loss, update='replace')
            
            step += 1
        # writer.flush()
        save_model_checkpoint(epoch=epoch, model=model, optimizer=optimizer,dataset_size=len(dataset))
        plot_dead_latents(dead_latents_per_epoch=dead_latents_per_epoch)
        activations = np.concatenate(activations, axis=0)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

        # Update Visdom plot for dead latents
        vis.line(Y=np.array([num_dead_latents]), X=np.array([epoch]), 
                 win=vis_dead_latents, update='append')
        # Create a new Visdom plot for the epoch's reconstruction losses
        vis.line(Y=np.array(epoch_losses), X=np.array(range(len(epoch_losses))),
                 opts=dict(title=f'Reconstruction Loss - Epoch {epoch+1}', 
                           xlabel='Step', ylabel='Loss'))

    print("\nEvaluating on Validation Set")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_embeddings = batch.to(device)
            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")

    print("\nEvaluating on Test Set")
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_embeddings = batch.to(device)
            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader)}")
    # writer.close()
    # tensorboard_process.terminate()
    # visdom_process.terminate()


if __name__ == "__main__":
    main()
