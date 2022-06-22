import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(

            # for clinical label only baseline
            nn.Linear(n_inputs, n_inputs),
            nn.LeakyReLU(0.2, True),
            nn.Linear(n_inputs, n_outputs)
        )

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        return X