import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_coef):
        super(SAE, self).__init__()
        self.encoder = self.encoder_initialize(input_dim, hidden_dim)
        self.decoder = self.decoder_initialize(input_dim, hidden_dim)
        self.lambda_coef = lambda_coef

    def encoder_initialize(self, input_dim, hidden_dim):
        linear = nn.Linear(input_dim, hidden_dim)
        relu = nn.ReLU()
        encoder = nn.Sequential(linear, relu)
        return encoder

    def decoder_initialize(self, input_dim, hidden_dim):
        linear = nn.Linear(hidden_dim, input_dim)
        return nn.Sequential(linear)

    def forward(self, x):
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


def loss_function(reconstructed, original, encoded, model):
    # Alternate approach is MAE, which is more robust to outliers but less used in the literature 
    mse_loss = nn.MSELoss()(reconstructed, original)
    kl_loss = model.l1_loss_function(encoded)
    return mse_loss + kl_loss
