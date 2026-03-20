import torch.nn as nn

from .linear_variational import LinearReparameterization


class DeterministicGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)
        self.gru3 = nn.GRU(hidden_size2, hidden_size3, batch_first=True)
        self.fc = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        return self.fc(out[:, -1, :])


class DeterministicLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        return self.fc(out[:, -1, :])


class VariationalGRUModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        out_features,
        prior_mean=0,
        prior_variance=1.0,
        posterior_rho_init=-3.0,
        bias=True,
    ):
        super().__init__()
        self.gru1 = nn.GRU(in_features, hidden_size1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)
        self.gru3 = nn.GRU(hidden_size2, hidden_size3, batch_first=True)
        self.fc = LinearReparameterization(
            in_features=hidden_size3,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias,
        )

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        hidden_last_step = out[:, -1, :]
        output, kl_fc = self.fc(hidden_last_step)
        return output, kl_fc


class VariationalLSTMModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        out_features,
        prior_mean=0,
        prior_variance=1.0,
        posterior_rho_init=-3.0,
        bias=True,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc = LinearReparameterization(
            in_features=hidden_size3,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias,
        )

    def forward(self, x, hidden_states=None):
        out, _ = self.lstm1(x, hidden_states)
        out, _ = self.lstm2(out, hidden_states)
        out, _ = self.lstm3(out, hidden_states)
        hidden_last_step = out[:, -1, :]
        output, kl_fc = self.fc(hidden_last_step)
        return output, kl_fc
