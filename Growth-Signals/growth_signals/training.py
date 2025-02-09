from tqdm import tqdm
from constants import HIDDEN_DIM, LR, BATCH_SIZE, EPOCHS
import torch.optim as optim
from torch.utils.data import DataLoader
from sae import SAE, loss_function
import torch
from custom_dataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

input_dim = dataset.embeddings.shape[1]  # Should be 768 for Cohere embeddings

model = SAE(input_dim, HIDDEN_DIM, lambda_coef=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        encoded, decoded = model(batch)
        loss = loss_function(decoded, batch, encoded, model)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(data_loader)}")
