import torch
import torch.nn as nn


class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """Calculates KL divergence between two Gaussians (Q || P)."""
        kl = torch.log(sigma_p) - torch.log(sigma_q)
        kl += (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2))
        kl -= 0.5
        return kl.mean()
