from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = torch.log(sigma_p) - torch.log(sigma_q) + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2 * sigma_p**2) - 0.5
        return kl.mean()


class LinearReparameterization(BaseVariationalLayer_):
    def __init__(
        self,
        in_features,
        out_features,
        prior_mean=0.0,
        prior_variance=1.0,
        posterior_mu_init=0.0,
        posterior_rho_init=-3.0,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias_flag = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu', torch.full((out_features, in_features), prior_mean), persistent=False)
        self.register_buffer('prior_weight_sigma', torch.full((out_features, in_features), prior_variance), persistent=False)

        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu', torch.full((out_features,), prior_mean), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.full((out_features,), prior_variance), persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self._kl = torch.tensor(0.0)
        self.reset_parameters()

    def reset_parameters(self):
        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)
        if self.bias_flag:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def forward(self, input):
        sigma_w = torch.log1p(torch.exp(self.rho_weight))
        eps_w = torch.randn_like(self.mu_weight)
        w = self.mu_weight + sigma_w * eps_w

        if self.bias_flag:
            sigma_b = torch.log1p(torch.exp(self.rho_bias))
            eps_b = torch.randn_like(self.mu_bias)
            b = self.mu_bias + sigma_b * eps_b
        else:
            b = None

        out = F.linear(input, w, b)

        kl_w = self.kl_div(self.mu_weight, sigma_w, self.prior_weight_mu, self.prior_weight_sigma)
        kl_b = torch.tensor(0.0, device=kl_w.device)
        if self.bias_flag:
            kl_b = self.kl_div(self.mu_bias, sigma_b, self.prior_bias_mu, self.prior_bias_sigma)
        self._kl = kl_w + kl_b
        return out

    @property
    def kl(self):
        return self._kl


class BaseBatteryModel(nn.Module):
    is_variational = False

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.config = config
        self.V0 = torch.tensor(config.V0, device=self.device, dtype=torch.float32)
        self.VEOD = torch.tensor(config.VEOD, device=self.device, dtype=torch.float32)
        self.Rp = torch.tensor(config.Rp, device=self.device, dtype=torch.float32)
        self.Rs = torch.tensor(config.Rs, device=self.device, dtype=torch.float32)
        self.Csp = torch.tensor(config.Csp, device=self.device, dtype=torch.float32)
        self.Cs = torch.tensor(config.Cs, device=self.device, dtype=torch.float32)
        self.init_x = self._build_init_state(config.batch_size)

    def _build_init_state(self, batch_size: int):
        return torch.tensor(self.config.x0, device=self.device, dtype=torch.float32).repeat(batch_size, 1)

    def set_batch_size(self, batch_size: int):
        self.init_x = self._build_init_state(batch_size)

    def dx(self, x, u):
        qb = x[:, 0].view(-1, 1)
        vs = x[:, 2].view(-1, 1) / self.Cs
        vsp = x[:, 1].view(-1, 1) / self.Csp

        soc = self.SOC(qb)
        rsp = self.g(soc.view(-1, 1))
        f_input = torch.cat([qb, soc], dim=1)
        vb = self.f(f_input) * self.V0

        vp = vb - vsp - vs
        ip = vp / self.Rp
        ib = u.view(-1, 1) + ip
        isp = ib - vsp * rsp
        i_s = ib - vs / self.Rs
        delta = torch.cat([-ib.view(-1, 1), isp.view(-1, 1), i_s.view(-1, 1)], dim=1)
        return delta

    def update_state(self, x, u):
        return x + self.dx(x, u)

    def output(self, x):
        qb = x[:, 0].view(-1, 1)
        vs = x[:, 2].view(-1, 1) / self.Cs
        vsp = x[:, 1].view(-1, 1) / self.Csp
        soc = self.SOC(qb)
        f_input = torch.cat([qb, soc], dim=1)
        vb = self.f(f_input) * self.V0
        return vb - vsp - vs

    def predict(self, input_seq):
        x = self.init_x
        state = [x]
        pred_v = []
        for n in range(input_seq.shape[1]):
            u = input_seq[:, n]
            x = self.update_state(x, u)
            output = self.output(x)
            pred_v.append(output.view(-1, 1))
            state.append(x)
        v_pred = torch.cat(pred_v, dim=1)
        state_tensor = torch.stack(state)
        return v_pred, state_tensor.transpose(0, 1)

    def boundary_loss(self, pred):
        upper = pred - self.V0
        lower = self.VEOD - pred
        return torch.mean(nn.ReLU()(upper)) + torch.mean(nn.ReLU()(lower))

    def kl_loss(self):
        return torch.tensor(0.0, device=self.V0.device)


class DeterministicBatteryModel(BaseBatteryModel):
    def __init__(self, config):
        super().__init__(config)
        self.SOC = nn.Sequential(nn.Linear(1, 4), nn.Linear(4, 1), nn.Sigmoid())
        self.f = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.g = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU(),
        )


class VariationalBatteryModel(BaseBatteryModel):
    is_variational = True

    def __init__(self, config):
        super().__init__(config)
        self.SOC = nn.Sequential(nn.Linear(1, 4), LinearReparameterization(4, 1), nn.Sigmoid())
        self.f = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            LinearReparameterization(8, 1),
            nn.Sigmoid(),
        )
        self.g = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            LinearReparameterization(4, 1),
            nn.ReLU(),
        )

    def kl_loss(self):
        kl_terms = [layer.kl for layer in self.modules() if hasattr(layer, 'kl')]
        if not kl_terms:
            return torch.tensor(0.0, device=self.V0.device)
        return sum(kl_terms)
