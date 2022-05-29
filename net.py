import torch.ao.ns.fx.utils
from torch import nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.conv_layer3 = self._conv_layer_set(64, 64)
        self.conv_layer4 = self._conv_layer_set(64, 32)

        self.fc1 = nn.Linear(6912, 3456)
        self.fc2 = nn.Linear(3456, 431)
        self.fc3 = nn.Linear(431, 256)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(3456)
        self.drop = nn.Dropout(p=0.15)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out

if __name__ == '__main__':
    net = CNNModel()
    model = torch.rand(9,3,128,128,128)
    output = net(model)
    print(output.shape)