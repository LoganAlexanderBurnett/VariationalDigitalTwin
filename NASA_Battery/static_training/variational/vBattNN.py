import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import os

class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = torch.log(sigma_p) - torch.log(sigma_q) \
             + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) \
             - 0.5
        return kl.mean()

class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean      = prior_mean
        self.prior_variance  = prior_variance
        self.posterior_mu_init  = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias_flag = bias

        # posterior params
        self.mu_weight  = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu',
                             torch.full((out_features, in_features), prior_mean),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.full((out_features, in_features), prior_variance),
                             persistent=False)

        if bias:
            self.mu_bias  = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu',
                                 torch.full((out_features,), prior_mean),
                                 persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.full((out_features,), prior_variance),
                                 persistent=False)
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
        eps_w   = torch.randn_like(self.mu_weight)
        w = self.mu_weight + sigma_w * eps_w

        if self.bias_flag:
            sigma_b = torch.log1p(torch.exp(self.rho_bias))
            eps_b   = torch.randn_like(self.mu_bias)
            b = self.mu_bias + sigma_b * eps_b
        else:
            b = None

        out = F.linear(input, w, b)

        kl_w = self.kl_div(self.mu_weight, sigma_w,
                           self.prior_weight_mu, self.prior_weight_sigma)
        kl_b = torch.tensor(0.0, device=kl_w.device)
        if self.bias_flag:
            kl_b = self.kl_div(self.mu_bias, sigma_b,
                               self.prior_bias_mu, self.prior_bias_sigma)
        self._kl = kl_w + kl_b
        return out

    @property
    def kl(self):
        return self._kl

class weighted_l1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.SmoothL1Loss(reduction='none', beta=0.1)
    def forward(self, inp, target, weight=None):
        seq = inp.shape[1]
        w = torch.linspace(seq/5,1,steps=seq, device=inp.device)
        mae = self.l1(inp, target)
        return torch.mean(mae * w)


class weighted_l2(nn.Module):
    def __init__(self):
        super(weighted_l2,self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self,input1,input2):
        seq_len = input1.shape[1]
        weight = torch.linspace(seq_len / 5, 1, steps=seq_len)

        mse = self.mse(input1,input2)
        weighted_mse = torch.mul(mse, weight)

        return torch.mean(weighted_mse)
        

class BattNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.config = config

        # fixed parameter
        #self.qMax = torch.tensor(config.qMax,device=self.device,dtype=torch.float32)
        self.V0 = torch.tensor(config.V0,device=self.device,dtype=torch.float32)
        #self.CMax = torch.tensor(config.CMax,device=self.device,dtype=torch.float32)
        self.init_x = torch.tensor(config.x0,device=self.device,dtype=torch.float32).repeat(config.batch_size,1) # [tb,qb,qcp,qcs]
        self.VEOD = torch.tensor(config.VEOD,device=self.device,dtype=torch.float32)

        self.Rp = torch.tensor(config.Rp,device=self.device,dtype=torch.float32) #10000
        self.Rs = torch.tensor(config.Rs,device=self.device,dtype=torch.float32) #0.0538926
        self.Csp = torch.tensor(config.Csp, device=self.device,dtype=torch.float32) #14.8223
        self.Cs = torch.tensor(config.Cs, device=self.device,dtype=torch.float32) #234.387

        if config.save_model is not None:
            self.save_name = f'results/BattNN-{config.save_model}-batch_size={config.batch_size}-seq_len={config.seq_len}.pkl'
        else:
            self.save_name = None

        # networks with variational layers
        self.SOC = nn.Sequential(
            nn.Linear(1,4),
            LinearReparameterization(4,1),
            nn.Sigmoid()
        )
        self.f = nn.Sequential(
            nn.Linear(2,8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            LinearReparameterization(8,1),
            nn.Sigmoid()
        )
        self.g = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            LinearReparameterization(4,1),
            nn.ReLU()
        )

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[100,500],gamma=0.5)

        #self.loss = nn.L1Loss()
        self.loss = weighted_l1()
        #self.loss = weighted_l2()
        #self.loss = nn.MSELoss()
        self.iter = 0

    def get_data(self, x, label):
        self.input = x.to(self.device)
        self.label = label.to(self.device)

    def dx(self,x,u):
        '''
        :param x: batch_size * [tb,qb,qcp,qcs]  [batch_size,4]
        :param u: batch_size * [i]              [batch_size,1]
        :return: dx
        '''
        qb = x[:,0].view(-1,1)
        Vs = x[:,2].view(-1,1) / self.Cs
        Vsp = x[:,1].view(-1,1) / self.Csp

        SOC = self.SOC(qb)
        Rsp = self.g(SOC.view(-1,1))
        f_input = torch.cat([qb,SOC],dim=1)
        Vb = self.f(f_input) * self.V0

        Vp = Vb - Vsp - Vs
        ip = Vp / self.Rp
        ib = u.view(-1,1) + ip
        isp = ib - Vsp * Rsp
        i_s = ib - Vs / self.Rs
        delta = torch.cat([-ib.view(-1,1),isp.view(-1,1),i_s.view(-1,1)],dim=1)

        return delta # [batch_size,4]

    def update_state(self,x,u):
        '''
        :param x: [batch_size,4]
        :param u: [batch_size,1]
        :return:
        '''
        dx = self.dx(x,u)  # [batch_size,4]
        new_x = x + dx
        return new_x

    def output(self,x):
        qb = x[:,0].view(-1,1)
        Vs = x[:,2].view(-1,1) / self.Cs
        Vsp = x[:,1].view(-1,1) / self.Csp
        SOC = self.SOC(qb)
        f_input = torch.cat([qb, SOC], dim=1)
        Vb = self.f(f_input) * self.V0

        V = Vb - Vsp - Vs  # [batch_size,1]
        return V

    def predict(self,input_seq):
        '''
        :param input_seq: [batch_size,seq_len]
        :return:
        '''
        x = self.init_x
        state = [x]
        pred_V = []
        steps = input_seq.shape[1]
        for n in range(steps):
            u = input_seq[:,n]   # [batch_size,1]
            x = self.update_state(x,u) # [batch_size,4]
            output = self.output(x)
            pred_V.append(output.view(-1,1))  # append: [batch_size,1]
            state.append(x)          # append: [batch_size,4]
        V_pred = torch.cat(pred_V,dim=1)
        State = torch.stack(state)
        return V_pred, State.transpose(0,1)

    def boundary_loss(self,pred):
        upper = pred - self.V0
        lowwer = self.VEOD - pred
        b_loss = torch.mean(nn.ReLU()(upper)) + torch.mean(nn.ReLU()(lowwer))
        return b_loss

    def train_one_epoch(self, print_per=50):
        self.optimizer.zero_grad()
        pred, _ = self.predict(self.input)
        l_1 = self.loss(pred, self.label)
        l_boundary = self.boundary_loss(pred)
        # KL sum
        kl = sum(layer.kl for layer in self.modules() 
                 if hasattr(layer, 'kl'))
        beta = getattr(self.config, 'kl_beta', 1e-5)
        loss = l_1 + l_boundary + beta * kl
        loss.backward()
        self.optimizer.step()
        self.iter += 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
        if self.iter % print_per==0 or self.iter==1:
            print(f"Epoch {self.iter:4d}  loss={loss:.4e}  "
                  f"L1={l_1:.7f}  bnd={l_boundary:.10f}  KL={kl:.7f}  lr={lr:.4f}")
        return loss
        

    def train(self):
        min_loss = 10
        stop = 0
        self.best_state = None
        LOSS = []
        for e in range(self.config.epoch):
            loss = self.train_one_epoch(print_per=50)
            self.scheduler.step()
            stop += 0
            LOSS.append(loss.item())
            if loss < min_loss:
                min_loss = loss.item()
                stop = 0
                self.best_state = {'net':self.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':self.iter}
            if stop > 50:
                print('\nearly stop')
                print('='*100)
                break
        self.save_model()
        return LOSS


    def save_model(self):
        # if self.save_name is not None:
        #     torch.save(self.best_state, self.save_name)
        if self.save_name is not None:
            # ensure parent directory exists
            parent = os.path.dirname(self.save_name)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            torch.save(self.best_state, self.save_name)
    
    def load_model(self):
        ckpt = torch.load(self.save_name)
        self.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.iter = ckpt['epoch'] + 1


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Battery Net for Simulate Data')
    parser.add_argument('--qMax',default=7856.3254)
    parser.add_argument('--V0',default=4.183)
    parser.add_argument('--CMax',default=7777)
    parser.add_argument('--x0',default=[7856.3254,0,0])
    parser.add_argument('--dt',default=1.0,help='length of time step')
    parser.add_argument('--VEOD',default=3.0)

    parser.add_argument('--Rp',default=10000)
    parser.add_argument('--Rs',default=0.0538926)
    parser.add_argument('--Csp',default=14.8223)
    parser.add_argument('--Cs',default=234.387)

    parser.add_argument('--batch_size',default=60)
    parser.add_argument('--seq_len',default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch',default=1000)
    parser.add_argument('--lr',default=2e-2)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--save_model',default='NASA',choices=[None,'NASA'])

    args = parser.parse_args()
    return args