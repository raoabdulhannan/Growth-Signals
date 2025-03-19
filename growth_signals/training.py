import os
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import visdom
import requests
import heapq
import json
import gc

from constants import HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, EMBEDDING_DIM, EPOCHS, LR
from custom_dataset import CustomDataset
from sae import SAE, loss_function
from plot import plot_dead_latents

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


# Function for saving data required for automated interpretability
def save_sae_data_for_interpretability(model, dataset, device, k=16, feature_count=None, save_dir='./sae_data'):
    os.makedirs(save_dir, exist_ok=True)
    
    if feature_count is None:
        feature_count = model.encoder[0].out_features
    
    topk_indices = np.zeros((feature_count, k), dtype=np.int32)
    topk_values = np.zeros((feature_count, k), dtype=np.float32)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, persistent_workers=True, pin_memory=True
    )
    
    doc_ids = []
    abstracts = []
    titles = []
    
    feature_heaps = [[] for _ in range(feature_count)]
    
    print("Collecting feature activations for interpretability...")
    global_idx = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            batch_embeddings, batch_texts, batch_titles = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            
            encoded, _ = model(batch_embeddings)
            encoded_np = encoded.cpu().numpy()
            
            for i, (emb, text, title) in enumerate(zip(encoded_np, batch_texts, batch_titles)):
                doc_ids.append(global_idx)
                abstracts.append(text)
                titles.append(title)
                
                for feature_idx in range(feature_count):
                    activation = emb[feature_idx]
                    
                    if len(feature_heaps[feature_idx]) < k:
                        heapq.heappush(feature_heaps[feature_idx], (activation, global_idx))
                    elif activation > feature_heaps[feature_idx][0][0]:
                        heapq.heappushpop(feature_heaps[feature_idx], (activation, global_idx))
                
                global_idx += 1
    
    for feature_idx in range(feature_count):
        sorted_activations = sorted(feature_heaps[feature_idx], reverse=True)
        
        for k_idx, (value, idx) in enumerate(sorted_activations):
            topk_indices[feature_idx, k_idx] = idx
            topk_values[feature_idx, k_idx] = value
    
    print(f"Saving SAE data to {save_dir}...")
    np.save(os.path.join(save_dir, f"topk_indices_{k}_{feature_count}.npy"), topk_indices)
    np.save(os.path.join(save_dir, f"topk_values_{k}_{feature_count}.npy"), topk_values)
    
    with open(os.path.join(save_dir, "abstract_texts.json"), 'w') as f:
        json.dump({"doc_ids": doc_ids, "abstracts": abstracts, "titles": titles}, f)
    
    print("SAE data saved successfully!")


def cleanup_gpu_memory():
    print("\nCleaning up GPU memory...")
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"GPU memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("Cleanup complete")

def main():
    print("\nTraining Final Model with Best Hyperparameters")
    
    # Wait for the Visdom server to start
    # wait_for_visdom()
    # vis = visdom.Visdom(port=8097)
    # vis_recon_loss = vis.line(
    #     Y=np.array([0]), X=np.array([0]),
    #     opts=dict(title='Reconstruction Loss',
    #               xlabel='Step', ylabel='Loss',
    #               width=800, height=400)
    # )
    # vis_dead_latents = vis.line(
    #     Y=np.array([0]), X=np.array([0]),
    #     opts=dict(title='Dead Latents',
    #               xlabel='Epoch', ylabel='Number of Dead Latents')
    # )
    
    dataset = CustomDataset(num_rows=1000000, shuffle=False)  # Default values: 100000 rows, no shuffling
    
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
    
    dead_latents_per_epoch = []
    step = 0
    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        total_loss = 0
        epoch_losses = []
        epoch_encoded_tensors = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch_embeddings, _, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'): # Changed from torch.cuda.amp.autocast()
                encoded, decoded = model(batch_embeddings)
                loss = loss_function(decoded, batch_embeddings, encoded, model)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_losses.append(min(loss.item(), 1))

            

            loss_value = min(loss.item(), 1)
            epoch_losses.append(loss_value)
            
            # Only accumulate activations every 10 steps to avoid frequent syncs
            if step % 10 == 0:
                epoch_encoded_tensors.append(encoded.detach())
                
            # Throttle Visdom updates to every 100 steps
            if step % 100 == 0:
                current_avg = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                # vis.line(
                #     Y=np.array([current_avg]),
                #     X=np.array([step]),
                #     win=vis_recon_loss,
                #     update='append'
                # )
            step += 1
        
        
        concat_encoded = torch.cat(epoch_encoded_tensors, dim=0)
        dead_latents = (concat_encoded == 0).all(dim=0)
        num_dead_latents = int(dead_latents.sum().item())
        dead_latents_per_epoch.append(num_dead_latents)
        print(f"Epoch {epoch+1} Dead Latents: {num_dead_latents}")
        
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss}")
        save_model_checkpoint(epoch=epoch, model=model, optimizer=optimizer, dataset_size=len(dataset))
        plot_dead_latents(dead_latents_per_epoch=dead_latents_per_epoch)

    
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
            batch_embeddings, _, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")
    
    print("\nEvaluating on Test Set")
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch_embeddings, _, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            encoded, decoded = model(batch_embeddings)
            loss = loss_function(decoded, batch_embeddings, encoded, model)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader)}")

    print("\nSaving SAE data for interpretability...")
    save_sae_data_for_interpretability(
        model=model, 
        dataset=dataset,
        device=device, 
        k=16,  # Top k examples saved per feature
        feature_count=HIDDEN_DIM,  # Total feature count
        save_dir='./sae_data'
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()