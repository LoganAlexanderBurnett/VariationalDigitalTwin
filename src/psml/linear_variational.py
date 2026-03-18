import torch
import torch.nn.functional as F
from torch.nn import Parameter

from .base_variational_layer import BaseVariationalLayer_


class LinearReparameterization(BaseVariationalLayer_):
    def __init__(
        self,
        in_features,
        out_features,
        prior_mean=0,
        prior_variance=1,
        posterior_mu_init=0,
        posterior_rho_init=-3.0,
        bias=True,
    ):
        super(LinearReparameterization, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = (posterior_mu_init,)
        self.posterior_rho_init = (posterior_rho_init,)
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_weight_mu', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_weight_sigma', torch.Tensor(out_features, in_features), persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_features), persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma,
        )
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias,
                sigma_bias,
                self.prior_bias_mu,
                self.prior_bias_sigma,
            )
        return kl

    def forward(self, input):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        weight = self.mu_weight + (sigma_weight * eps_weight)

        kl_weight = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma,
        )

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            kl_bias = self.kl_div(
                self.mu_bias,
                sigma_bias,
                self.prior_bias_mu,
                self.prior_bias_sigma,
            )

        out = F.linear(input, weight, bias)
        kl = kl_weight + kl_bias if self.mu_bias is not None else kl_weight
        return out, kl
