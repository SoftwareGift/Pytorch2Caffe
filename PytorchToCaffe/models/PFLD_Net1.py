import torch.nn as nn
import math
from .Auxiliary import Auxiliary
import torch


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = int(expansion * in_planes)
    
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.Relu6 = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.Relu6(self.bn1(self.conv1(x)))
        out = self.Relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class PFLD_Net(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(2, 64, 5, 2),
           (2, 128, 1, 2),
           (4, 128, 6, 1),
           (2, 16, 1, 1)]

    def __init__(self, net_size):
        super(PFLD_Net, self).__init__()
        self.net_size = net_size
        input_channel=int(64*net_size)

        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)

        self.conv2 = nn.Conv2d(input_channel, input_channel, kernel_size=3,
                               stride=1, padding=1, groups=input_channel, bias=False)
        self.bn2 = nn.BatchNorm2d(input_channel)
        self.conv3 = nn.Conv2d(input_channel, input_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(input_channel)

        self.first_block = self._make_first_block(in_planes=input_channel)
        self.blocks = self._make_blocks(in_planes=input_channel)

        self.conv4 = nn.Conv2d(int(16*net_size), int(32*net_size), kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(32*net_size))
        self.conv5 = nn.Conv2d(int(32*net_size), int(128*net_size), kernel_size=7,
                               stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(int(128*net_size))
        self.Relu6 = nn.ReLU6(inplace=True)
        self.fc = nn.Linear(int(4832*net_size), 136)

        self.pose = Auxiliary(net_size=net_size)
        self._initialize_weights()

    def _make_first_block(self, in_planes):
        expansion, out_planes, num_blocks, stride = self.cfg[0]
        out_planes = int(out_planes*self.net_size)
        strides = [stride] + [1]*(num_blocks-1)
        layers = list()
        for stride in strides:
            layers.append(Block(in_planes, out_planes, expansion, stride))
        return nn.Sequential(*layers)

    def _make_blocks(self, in_planes):
        layers = []
        for id_x, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg[1:]):
            out_planes = int(out_planes*self.net_size)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.Relu6(self.bn1(self.conv1(x)))
        out = self.Relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.first_block(out)
        S1 = self.blocks(out)
        S2 = self.Relu6(self.bn4(self.conv4(S1)))
        S3 = self.Relu6(self.bn5(self.conv5(S2)))
        S1 = S1.view(S1.size(0), -1)
        S2 = S2.view(S2.size(0), -1)
        S3 = S3.view(S3.size(0), -1)
        landmark = torch.cat([S1, S2, S3], 1)
        landmark = self.fc(landmark)
        landmark = landmark.view(landmark.size(0), 68, 2)
        pose = self.pose(out)
        return landmark, pose

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
    net = PFLD_Net(0.25)
    import torch
    # x = torch.randn(2, 3, 112, 112).cuda()
    # y = net(x)
    # print(y)
    net = net.to('cuda')
    from torchsummary import summary
    summary(net, (3, 112, 112))
