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
import matplotlib.pyplot as plt

from constants import HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, EMBEDDING_DIM, EPOCHS, LR, LAMBDA, SPARSITY_START_EPOCH
from plot_latent_space import plot_re_ranked_vectors
from custom_dataset import CustomDataset
from sae import SAE, loss_function, reward_function
from plot import plot_dead_latents
from plot_latent_space import plot_re_ranked_vectors

USE_REWARD = True
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
def save_sae_data_for_interpretability(model, dataset, device, top_k=10, zero_k=20, random_k=10, feature_count=None, save_dir='./sae_data'):
    os.makedirs(save_dir, exist_ok=True)
    
    if feature_count is None:
        feature_count = model.encoder[0].out_features
    
    topk_indices = np.zeros((feature_count, top_k), dtype=np.int32)
    topk_values = np.zeros((feature_count, top_k), dtype=np.float32)
    
    zero_indices = np.zeros((feature_count, zero_k), dtype=np.int32)
    zero_similarities = np.zeros((feature_count, zero_k), dtype=np.float32)
    
    random_indices = np.zeros((feature_count, random_k), dtype=np.int32)
    random_values = np.zeros((feature_count, random_k), dtype=np.float32)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, persistent_workers=True, pin_memory=True
    )
    
    doc_ids = []
    abstracts = []
    titles = []
    embeddings = []
    
    feature_heaps = [[] for _ in range(feature_count)]
    feature_activations = {i: [] for i in range(feature_count)}
    
    print("Collecting feature activations for interpretability...")
    global_idx = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            batch_embeddings, batch_texts, batch_titles = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            
            encoded, _ = model(batch_embeddings)
            encoded_np = encoded.cpu().numpy()
            
            for i, (emb, text, title, orig_emb) in enumerate(zip(encoded_np, batch_texts, batch_titles, batch_embeddings.cpu().numpy())):
                doc_ids.append(global_idx)
                abstracts.append(text)
                titles.append(title)
                embeddings.append(orig_emb)
                
                for feature_idx in range(feature_count):
                    activation = emb[feature_idx]
                    feature_activations[feature_idx].append((activation, global_idx))
                    
                    if len(feature_heaps[feature_idx]) < top_k:
                        heapq.heappush(feature_heaps[feature_idx], (activation, global_idx))
                    elif activation > feature_heaps[feature_idx][0][0]:
                        heapq.heappushpop(feature_heaps[feature_idx], (activation, global_idx))
                
                global_idx += 1
    
    embeddings = np.array(embeddings)
    
    print("Finding low-activating and random examples...")
    for feature_idx in tqdm(range(feature_count), desc="Processing features"):

        sorted_activations = sorted(feature_heaps[feature_idx], reverse=True)
        
        for k_idx, (value, idx) in enumerate(sorted_activations):
            topk_indices[feature_idx, k_idx] = idx
            topk_values[feature_idx, k_idx] = value
        
        top_indices = [idx for _, idx in sorted_activations]
        top_embeddings = embeddings[top_indices]
        
        norms = np.linalg.norm(top_embeddings, axis=1, keepdims=True)
        normalized_embeddings = top_embeddings / norms
        
        mean_embedding = np.mean(normalized_embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        
        all_activations = [(act, idx) for act, idx in feature_activations[feature_idx]]
        
        sorted_by_activation = sorted(all_activations)
        
        exact_zeros = [(act, idx) for act, idx in all_activations if act == 0]
        if len(exact_zeros) > 0:
            print(f"Feature {feature_idx} has {len(exact_zeros)} exact zero activations")
        
        zero_act_pairs = sorted_by_activation[:zero_k * 3]
        
        if len(zero_act_pairs) > zero_k:
            indices = np.random.choice(len(zero_act_pairs), size=zero_k, replace=False)
            zero_act_pairs = [zero_act_pairs[i] for i in indices]
            
        zero_act_indices = [idx for _, idx in zero_act_pairs]
        
        zero_act_embeddings = embeddings[zero_act_indices]
        norms = np.linalg.norm(zero_act_embeddings, axis=1, keepdims=True)
        normalized_embeddings = zero_act_embeddings / norms
        
        similarities = np.dot(normalized_embeddings, mean_embedding)
        
        sorted_indices = np.argsort(-similarities)
        
        for idx, sim_idx in enumerate(sorted_indices):
            if idx < zero_k:
                zero_indices[feature_idx, idx] = zero_act_indices[sim_idx]
                zero_similarities[feature_idx, idx] = similarities[sim_idx]
        
        avg_activation = np.mean([act for act, _ in all_activations if act > 0])
        non_zero_pairs = [(act, idx) for act, idx in all_activations 
                          if act > avg_activation * 0.5 and idx not in top_indices]
        
        if len(non_zero_pairs) < random_k:
            print(f"Warning: Only {len(non_zero_pairs)} significant non-zero activating examples found for feature {feature_idx}")
            non_zero_pairs = [(act, idx) for act, idx in all_activations 
                             if act > 0 and idx not in top_indices]
            
        if len(non_zero_pairs) > 0:
            random_samples = np.random.choice(len(non_zero_pairs), size=min(random_k, len(non_zero_pairs)), replace=False)
            for idx, sample_idx in enumerate(random_samples):
                act, doc_idx = non_zero_pairs[sample_idx]
                random_indices[feature_idx, idx] = doc_idx
                random_values[feature_idx, idx] = act
        else:
            print(f"Warning: No positive activating examples found for feature {feature_idx} beyond top-k")
    
    print(f"Saving SAE data to {save_dir}...")
    np.save(os.path.join(save_dir, f"topk_indices_{top_k}_{feature_count}.npy"), topk_indices)
    np.save(os.path.join(save_dir, f"topk_values_{top_k}_{feature_count}.npy"), topk_values)
    np.save(os.path.join(save_dir, f"zero_indices_{zero_k}_{feature_count}.npy"), zero_indices)
    np.save(os.path.join(save_dir, f"zero_similarities_{zero_k}_{feature_count}.npy"), zero_similarities)
    np.save(os.path.join(save_dir, f"random_indices_{random_k}_{feature_count}.npy"), random_indices)
    np.save(os.path.join(save_dir, f"random_values_{random_k}_{feature_count}.npy"), random_values)
    
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
    
    dataset = CustomDataset(num_rows=100000, shuffle=False)  # Default values: 100000 rows, no shuffling
    
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

                if USE_REWARD:
                    sparsity_mask = (encoded != 0).float()
                    reward = reward_function(
                        reconstructed=decoded,
                        original=batch_embeddings,
                        sparsity_mask=sparsity_mask,
                        lambda_sparsity=0.01
                    )
                    loss = -reward
                else:
                    apply_l1 = epoch >= SPARSITY_START_EPOCH
                    loss = loss_function(decoded, batch_embeddings, encoded, model, apply_l1=apply_l1)

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
        if USE_REWARD:
            print(f"Epoch {epoch+1}/{EPOCHS}, Reward (neg_loss): {-avg_epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")

        save_model_checkpoint(epoch=epoch, model=model, optimizer=optimizer, dataset_size=len(dataset))
        # plot_dead_latents(dead_latents_per_epoch=dead_latents_per_epoch)

    print("\nEvaluating on Validation Set")
    val_score = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch_embeddings, _, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            encoded, decoded = model(batch_embeddings)

            if USE_REWARD:
                sparsity_mask = (encoded != 0).float()
                score = reward_function(
                    reconstructed=decoded,
                    original=batch_embeddings,
                    sparsity_mask=sparsity_mask,
                    lambda_sparsity=LAMBDA
                )
            else:
                score = -loss_function(decoded, batch_embeddings, encoded, model)

            val_score += score.item()
    print(f"Avg Validation {'Reward' if USE_REWARD else 'Loss'}: {val_score / len(val_loader):.4f}")

    print("\nEvaluating on Test Set")
    test_score = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch_embeddings, _, _ = batch
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            encoded, decoded = model(batch_embeddings)

            if USE_REWARD:
                sparsity_mask = (encoded != 0).float()
                score = reward_function(
                    reconstructed=decoded,
                    original=batch_embeddings,
                    sparsity_mask=sparsity_mask,
                    lambda_sparsity=LAMBDA
                )
            else:
                score = -loss_function(decoded, batch_embeddings, encoded, model)

            test_score += score.item()
    print(f"Avg Test {'Reward' if USE_REWARD else 'Loss'}: {test_score / len(test_loader):.4f}")


    print("\nSaving SAE data for interpretability...")
    save_sae_data_for_interpretability(
        model=model, 
        dataset=dataset,
        device=device, 
        top_k=10,
        zero_k=20,
        random_k=10,
        feature_count=HIDDEN_DIM,
        save_dir='./sae_data'
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()