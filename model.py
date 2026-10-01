import torch
import torch.nn as nn

## MLP takes as input the noised-up data pair (x_t, y) at time-step t and returns a prediction of the noise at that time-step
class MLP(nn.Module):
    def __init__(self, n_steps, input_dim, output_dim):
        super().__init__()

        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.01),
            nn.GELU()
        )
        self.embedding_layer = nn.Embedding(n_steps, 256)
        
        self.linear_model2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.01),
            nn.GELU(),
            
            nn.Linear(512, 512),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.01),
            nn.GELU(),
            
            nn.Linear(512, 256),
        )

        self.linear_model3 = nn.Sequential(
            nn.Linear(256, output_dim)
        )
    def forward(self, x, t):
        x1 = self.linear_model1(x)
        x2 = self.linear_model2(x1 + self.embedding_layer(t))
        x = self.linear_model3(x2 + x1)
        return x


class Model(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim, output_dim):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : diffusion steps
        input_dim : dimension of data

        '''
        super().__init__()
        self.device = device
        # beta's are the schedule with which the noise is added to the data
        # alpha_bars is cumulative product of alpha_bar (1 - beta_1) * (1 - beta_2) * ... * (1 - beta_T)
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        # mlp predicts the noise based on x_tilde, which is data with noise added
        self.mlp = MLP(T, input_dim, output_dim)
        
        self.to(device = self.device)

    def loss_fn(self, x, y, t=None):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if None (training phase), we add noise at random timesteps. 
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.

        '''
        # output comes from forward process, value calculated in backbone
        # epsilon is a sampled random parameter that determines the noise added onto the data (vector of random numbers shaped like data)
        # alpha's are like normalization factors that are used to scale the noise
        output, epsilon = self.forward(x, y, t=t, train=True)
        loss = (output - epsilon).square().mean()
        return loss

        
    def forward(self, x, y, t, train):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if training phase, we add noise at random timesteps. 
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.
        train      : if True (training phase), target is returned along with epsilon prediction

        '''

        if train:
            # this whole thing adds noise, this is the whole forward process
            epsilon = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.randint(0, len(self.alpha_bars), (len(x), )).to(device = self.device)
            # used_alpha_bar is the cumulative product of alpha's used up to time t in this iteration
            used_alpha_bars = self.alpha_bars[t][:, None]

            # x_t is x with noise added
            x_t = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
            
        else: # if not in training, we don't need to add noise, because we are inputting a pure-noise vector
            x_t = x
            t = torch.tensor([t]).repeat(x.size(0)).to(self.device).long()


        
        noised_pair = torch.cat((x_t, y), dim = 1)
        
        # output is the prediction of the noise epsilon
        output = self.mlp(noised_pair, t)
        
        return (output, epsilon) if train else output