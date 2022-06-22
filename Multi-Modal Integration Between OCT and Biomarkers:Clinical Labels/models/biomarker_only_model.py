import torch.nn as nn

# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(

            # for att_only
            nn.Linear(n_inputs, 15),
            nn.LeakyReLU(0.2, True),
            nn.Linear(15, 10),
            nn.LeakyReLU(0.2, True),
            nn.Linear(10, 5),
            nn.LeakyReLU(0.2, True),
            nn.Linear(5, n_outputs)

        )

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        return X
