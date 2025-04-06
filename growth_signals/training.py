import os
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import visdom
import requests
import matplotlib.pyplot as plt

from constants import HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, EMBEDDING_DIM, EPOCHS, LR
from custom_dataset import CustomDataset
from sae import SAE, loss_function
from plot import plot_dead_latents
from plot_latent_space import plot_re_ranked_vectors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


"""
For running Visdom:
1. Install: pip install visdom
2. Run: python -m visdom.server in a separate terminal
3. Run the current script
(vis = visdom.Visdom(port=8097) will connect to the server you opened in step 2)
(All required code are already uncommented)
"""

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

# def wait_for_visdom():
#     visdom_url = "http://localhost:8097"
#     while True:
#         try:
#             requests.get(visdom_url)
#             print("Visdom server is up and running.")
#             break
#         except requests.ConnectionError:
#             print("Waiting for Visdom server...")
#             time.sleep(1)

def main():
    print("\nTraining Final Model with Best Hyperparameters")
    
    # Wait for the Visdom server to start
    # wait_for_visdom()
    # vis = visdom.Visdom(port=8097)
    # vis_loss_plot = vis.line(
    #     Y=np.array([0]), X=np.array([0]),
    #     opts=dict(title='Training and Test Loss',
    #               xlabel='Step', ylabel='Loss',
    #               legend=['Training Loss', 'Test Loss'],
    #               width=800, height=400))
    # vis_dead_latents = vis.line(
    #     Y=np.array([0]), X=np.array([0]),
    #     opts=dict(title='Dead Latents',
    #               xlabel='Epoch', ylabel='Number of Dead Latents')
    # )
    
    dataset = CustomDataset()
    
    # Split dataset into train/validation/test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print("\nCustom Dataset Loaded")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4,
                              persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4,
                            persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4,
                             persistent_workers=True, pin_memory=True)
    test_iter = iter(test_loader)
    print(f"\nData processed: {type(train_loader)}")
    
    model = SAE(
        input_dim=768,
        hidden_dim=HIDDEN_DIM,
        lambda_coef=0.1,
        decoder_activation="sigmoid",
        device=device
    ).to(device)
    model = torch.compile(model)  # Optional: requires PyTorch 2.0+
    print("\nModel loaded")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("\nOptimizer loaded")

    sample_indices = torch.randperm(len(train_dataset))[:10]
    sample_data = [train_dataset[i] for i in sample_indices]
    sample_embeddings = torch.stack([data[0] for data in sample_data]).to(device)
    sample_titles = [data[2] for data in sample_data]
    
    dead_latents_per_epoch = []
    step = 0
    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        total_loss = 0
        epoch_losses = []
        epoch_test_losses = []
        epoch_encoded_tensors = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch_embeddings, _, batch_titles = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                encoded, decoded = model(batch_embeddings)
                loss = loss_function(decoded, batch_embeddings, encoded, model)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_losses.append(loss.item())

            # Evaluate on a batch from the test set
            try:
                test_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_batch = next(test_iter)
            test_embeddings, _, _ = test_batch
            test_embeddings = test_embeddings.to(device, non_blocking=True)
            with torch.no_grad():
                test_encoded, test_decoded = model(test_embeddings)
                test_loss = loss_function(test_decoded, test_embeddings, test_encoded, model).item()
                epoch_test_losses.append(test_loss)

            # # Update Visdom plot for reconstruction loss
            # if step > 1000:
            #     vis.line(Y=np.array([epoch_losses[-1000:], epoch_test_losses[-1000:]]),
            #              X=np.array(range(step-999, step+1)),
            #              win=vis_loss_plot, update='replace')
            # else:
            #     vis.line(Y=np.array([epoch_losses, epoch_test_losses]),
            #              X=np.array(range(step-len(epoch_losses)+1, step+1)),
            #              win=vis_loss_plot, update='replace')

            # Only accumulate activations every 10 steps to avoid frequent syncs
            if step % 10 == 0:
                epoch_encoded_tensors.append(encoded.detach())
                
            # Throttle Visdom updates to every 100 steps
            # if step % 100 == 0:
            #     current_avg = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                # vis.line(
                #     Y=np.array([current_avg]),
                #     X=np.array([step]),
                #     win=vis_recon_loss,
                #     update='append'
                # )
            step += 1
        
        with torch.no_grad():
            sample_encoded, _ = model(sample_embeddings)
            plot_re_ranked_vectors(sample_encoded, sample_titles, epoch + 1, step)

        # Save training and testing loss plot
        os.makedirs("loss_plots", exist_ok=True)
        plt.figure()
        plt.plot(range(len(epoch_losses)), epoch_losses, label="Training Loss", color="blue")
        plt.plot(range(len(epoch_test_losses)), epoch_test_losses, label="Test Loss", color="orange")
        plt.title(f"Epoch {epoch+1} Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_plots/epoch_{epoch+1}_loss.png")
        plt.close()
        
        concat_encoded = torch.cat(epoch_encoded_tensors, dim=0)
        dead_latents = (concat_encoded == 0).all(dim=0)
        num_dead_latents = int(dead_latents.sum().item())
        dead_latents_per_epoch.append(num_dead_latents)
        print(f"Epoch {epoch+1} Dead Latents: {num_dead_latents}")
        
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss}")
        save_model_checkpoint(epoch=epoch, model=model, optimizer=optimizer, dataset_size=len(dataset))
        # plot_dead_latents(dead_latents_per_epoch=dead_latents_per_epoch)

    
        # vis.line(
        #     Y=np.array([num_dead_latents]),
        #     X=np.array([epoch]),
        #     win=vis_dead_latents,
        #     update='append'
        # )
    
    print(dead_latents_per_epoch)
    print("\nEvaluating on Validation Set")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch_embeddings, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")
    
    print("\nEvaluating on Test Set")
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch_embeddings, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader)}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
