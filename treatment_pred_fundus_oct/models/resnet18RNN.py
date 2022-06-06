from torch import nn
import torchvision.models as models

class Resnt18Rnn(nn.Module):
    def __init__(self, num_classes=2,dr_rate=.1,pretrained=False,rnn_num_layers=1,rnn_hidden_size=100):
        super(Resnt18Rnn, self).__init__()
        num_classes = num_classes
        dr_rate = dr_rate
        pretrained = pretrained
        rnn_hidden_size = rnn_hidden_size
        rnn_num_layers = rnn_num_layers

        baseModel = models.resnet18(pretrained=pretrained)
        baseModel.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape

        ii = 0
        y = self.baseModel((x[:, ii]))

        k = y.unsqueeze(1)

        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:, ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(output[:, -1])
        out = self.fc1(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x