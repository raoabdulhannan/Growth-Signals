from tqdm import tqdm
from constants import EPOCHS
import torch.optim as optim
from torch.utils.data import DataLoader
from sae import SAE, loss_function
import torch
from dataset import CustomDataset
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

search_space = [
    Real(1e-5, 1e-3, "log-uniform", name="learning_rate"),
    Integer(32, 128, 256, name="batch_size"),
    Real(0.01, 0.1, 0.5, name="lambda_coef"),
    Integer(128, 256, 512, 1536, name="hidden_dim"),
    Categorical(["identity", "sigmoid", "tanh"], name="decoder_activation")
]


def train_and_evaluate(params):
    lr, batch_size, lambda_coef, hidden_dim, decoder_activation = params

    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


print("\n Bayesian Optimization for Hyperparameter Tuning")
result = gp_minimize(train_and_evaluate, search_space, n_calls=20,
                     random_state=42)

best_params = result.x
best_config = {
    "learning_rate": best_params[0],
    "batch_size": best_params[1],
    "lambda_coef": best_params[2],
    "hidden_dim": best_params[3],
    "decoder_activation": best_params[4]
}

print(f"\n Best Hyperparameters: {best_config}")

print("\n Training Final Model with Best Hyperparameters")
dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=best_config["batch_size"],
                         shuffle=True)

model = SAE(
    input_dim=768,
    hidden_dim=best_config["hidden_dim"],
    lambda_coef=best_config["lambda_coef"],
    decoder_activation=best_config["decoder_activation"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = batch.to(device)
        optimizer.zero_grad()

        encoded, decoded = model(batch)
        loss = loss_function(decoded, batch, encoded, model)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(data_loader)}")
