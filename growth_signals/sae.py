import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_coef, decoder_activation, device=None):
        super(SAE, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "identity": nn.Identity()}[decoder_activation]
        )
        self.lambda_coef = lambda_coef

    def encoder_initialize(self, input_dim, hidden_dim):
        linear = nn.Linear(input_dim, hidden_dim)
        relu = nn.ReLU()
        encoder = nn.Sequential(linear, relu)
        return encoder

    def decoder_initialize(self, input_dim, hidden_dim, decoder_activation):
        linear = nn.Linear(hidden_dim, input_dim)
        if decoder_activation == "sigmoid":
            activation = nn.Sigmoid()
        elif decoder_activation == "tanh":
            activation = nn.Tanh()
        elif decoder_activation == "identity":
            activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {decoder_activation}")

        return nn.Sequential(linear, activation)

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def l1_loss_function(self, encoded):
        # L1 regularization for now, other option is KL divergence
        return self.lambda_coef * torch.norm(encoded, p=1)

    def kl_divergence(self, encoded):
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
        kl_loss = torch.sum(
            rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )
        return self.lambda_coef * kl_loss


def loss_function(reconstructed, original, encoded, model, apply_l1=True):
    mse_loss = nn.MSELoss()(reconstructed, original)
    if apply_l1:
        l1_loss = model.l1_loss_function(encoded)
        return mse_loss + l1_loss
    else:
        return mse_loss


def reward_function(reconstructed, original, sparsity_mask, lambda_sparsity):
    mse_loss = nn.MSELoss()(reconstructed, original)
    sparsity_penalty = lambda_sparsity * torch.sum(sparsity_mask)  # Fewer active neurons = better
    diversity_loss = -torch.var(sparsity_mask)  # Encourage diverse activation patterns

    reward = -mse_loss - sparsity_penalty + diversity_loss  # Maximize reward
    return reward
