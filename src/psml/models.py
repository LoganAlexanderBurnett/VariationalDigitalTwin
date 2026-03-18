import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_variational import LinearReparameterization


class StandardGRUModel(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size, bias=bias)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, out_features, bias=bias)

    def forward(self, x, hidden_states=None):
        x = F.relu(self.fc1(x))
        if hidden_states is None:
            hidden_states = torch.zeros(
                self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device
            )
        hidden_seq, _ = self.gru(x, hidden_states)
        hidden_last = F.relu(self.fc2(hidden_seq[:, -1, :]))
        return self.fc3(hidden_last)


class StandardLSTMModel(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size, bias=bias)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, out_features, bias=bias)

    def forward(self, x, hidden_states=None):
        x = F.relu(self.fc1(x))
        if hidden_states is None:
            hidden_states = (
                torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device),
                torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device),
            )
        hidden_seq, _ = self.lstm(x, hidden_states)
        hidden_last = F.relu(self.fc2(hidden_seq[:, -1, :]))
        return self.fc3(hidden_last)


class GRUReparameterizationModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_features,
        num_layers,
        prior_mean=0,
        prior_variance=1.0,
        posterior_rho_init=-3.0,
        bias=True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size, bias=bias)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = LinearReparameterization(
            in_features=hidden_size,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias,
        )

    def forward(self, x, hidden_states=None):
        x = F.relu(self.fc1(x))
        _, h_n = self.gru(x, hidden_states)
        last_hidden = F.relu(self.fc2(h_n[-1]))
        output, kl = self.fc3(last_hidden)
        return output, kl


class LSTMReparameterizationModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_features,
        num_layers,
        prior_mean=0,
        prior_variance=0.5,
        posterior_rho_init=-4.0,
        bias=True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size, bias=bias)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = LinearReparameterization(
            in_features=hidden_size,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias,
        )

    def forward(self, x, hidden_states=None):
        _, (h_n, _) = self.lstm(F.relu(self.fc1(x)), hidden_states)
        last_hidden = F.relu(self.fc2(h_n[-1]))
        output, kl = self.fc3(last_hidden)
        return output, kl


class RollingStandardGRUModel(nn.Module):
    def __init__(self, in_f, h, out_f, nl):
        super().__init__()
        self.fc1 = nn.Linear(in_f, h)
        self.gru = nn.GRU(h, h, num_layers=nl, batch_first=True)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, out_f)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        if hidden is None:
            hidden = torch.zeros(
                self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device
            )
        seq, h_n = self.gru(x, hidden)
        last = F.relu(self.fc2(seq[:, -1, :]))
        return self.fc3(last), h_n


class RollingStandardLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        seq, (h_n, c_n) = self.lstm(x, hidden)
        last = F.relu(self.fc2(seq[:, -1, :]))
        out = self.fc3(last)
        return out, (h_n, c_n)
