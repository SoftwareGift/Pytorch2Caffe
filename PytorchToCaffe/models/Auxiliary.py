import torch.nn as nn
import math


class Conv_Bn(nn.Module):
    def __init__(self, inp, outp, kernel_size, stride, padding):
        super(Conv_Bn, self).__init__()
        self.conv = nn.Conv2d(inp, outp, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        return x


class Auxiliary(nn.Module):
    def __init__(self, net_size=1.):
        super(Auxiliary, self).__init__()
        self.conv_bn1 = Conv_Bn(int(64*net_size), int(128*net_size), 3, 2, 1)
        self.conv_bn2 = Conv_Bn(int(128*net_size), int(128*net_size), 3, 1, 1)
        self.conv_bn3 = Conv_Bn(int(128*net_size), int(32*net_size), 3, 2, 1)
        self.conv_bn4 = Conv_Bn(int(32*net_size), int(128*net_size), 7, 1, 0)
        self.fc1 = nn.Linear(int(128*net_size), int(32*net_size))
        self.fc2 = nn.Linear(int(32*net_size), 3)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    net = Auxiliary()
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    from torchsummary import summary
    summary(net, (64, 28, 28))
