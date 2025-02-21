from datasets import load_dataset
from tqdm import tqdm
from constants import EPOCHS
import torch.optim as optim
from torch.utils.data import DataLoader
from sae import SAE, loss_function
import torch
from custom_dataset import CustomDataset
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

search_space = [
    Real(1e-5, 1e-3, "log-uniform", name="learning_rate"),
    # Categorical([32, 64, 128], name="batch_size"),
    Real(0.02, 0.08, "log-uniform", name="lambda_coef"),
    Categorical([1536, 2304], name="hidden_dim"),
    Categorical(["identity", "sigmoid", "tanh"], name="decoder_activation")
]


def train_and_evaluate(params):
    lr, lambda_coef, hidden_dim, decoder_activation = params
    batch_size = 32
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
result = gp_minimize(train_and_evaluate, search_space, n_calls=10,
                     random_state=42)

best_params = result.x
print(f"Best params are :{best_params}")
best_config = {
    "learning_rate": best_params[0],
    "lambda_coef": best_params[1],
    "hidden_dim": best_params[2],
    "decoder_activation": best_params[3]
}

print(f"\n Best Hyperparameters: {best_config}")

