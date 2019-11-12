import torch.nn as nn
import math
import torch


class Conv(nn.Module):
    def __init__(self, inp, outp, ks, s=1, p=0, gs=1, relu='relu'):
        r"""input,output,kernel_size,stride,padding,groups
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inp, outp, ks, s, p, groups=gs, bias=False)
        self.bn = nn.BatchNorm2d(outp)
        if relu == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif relu == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu != None:
            x = self.relu(x)
        return x


class Auxiliary(nn.Module):
    def __init__(self, net_size=1.):
        super(Auxiliary, self).__init__()
        scale = int(1//net_size)
        self.conv1 = Conv(64//scale, 128//scale, 3, 2, 1, relu='relu')
        self.conv2 = Conv(128//scale, 128//scale, 3, 1, 1, relu='relu')
        self.conv3 = Conv(128//scale, 32//scale, 3, 2, 1, relu='relu')
        self.conv4 = Conv(32//scale, 128//scale, 7, 1, 0, relu='relu')
        self.fc1 = nn.Linear(128//scale, 32//scale)
        self.fc2 = nn.Linear(32//scale, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_c, out_c, t, stride, downsample=False):
        super(Block, self).__init__()
        t_c = int(in_c*t)
        self.conv1 = Conv(in_c, t_c, 1, 1, 0, relu='relu6')
        self.conv2 = Conv(t_c, t_c, 3, stride, 1, gs=t_c, relu='relu6')
        self.conv3 = Conv(t_c, out_c, 1, 1, 0, relu=None)
        self.downsample = downsample
        self.shortcut=None
        if in_c!=out_c:
            self.shortcut=Conv(in_c,out_c,1,1,0,relu=None)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if not self.downsample:
            if self.shortcut!=None:
                out+=self.shortcut(x)
            else:
                out+=x
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
        scale = int(1//net_size)
        in_c = 64//scale
        self.conv1 = Conv(3, in_c, 3, 2, 1, relu='relu')

        # dw pw
        self.conv2 = Conv(in_c, in_c, 3, 1, 1, in_c, relu=None)
        self.conv3 = Conv(in_c, in_c, 1, 1, 0, relu=None)

        self.block1 = self._make_block(in_c, [2, 64//scale, 5, 2])
        self.block2 = self._make_block(64//scale, [2, 128//scale, 1, 2])
        self.block3 = self._make_block(128//scale, [4, 128//scale, 6, 1])
        self.block4 = self._make_block(128//scale, [2, 16//scale, 1, 1])

        self.conv4 = Conv(16//scale, 32//scale, 3, 2, 1, relu='relu')
        self.conv5 = Conv(32//scale, 128//scale, 7, 1, 0, relu='relu')

        self.fc = nn.Linear(4832//scale, 136)
        self.pose = Auxiliary(net_size=net_size)
        self._initialize_weights()

    def _make_block(self, in_c, cfg):
        t, out_c, num_blocks, s = cfg
        layers = []
        if s == 2:
            layers.append(Block(in_c, out_c, t, stride=2, downsample=True))
            in_c = out_c
            num_blocks -= 1
        for _ in range(num_blocks):
            layers.append(Block(in_c, out_c, t, stride=1, downsample=False))
            in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.block1(x)
        out = self.block2(x)
        out = self.block3(out)
        S1 = self.block4(out)
        S2 = self.conv4(S1)
        S3 = self.conv5(S2)
        S1 = S1.view(S1.size(0), -1)
        S2 = S2.view(S2.size(0), -1)
        S3 = S3.view(S3.size(0), -1)
        landmark = torch.cat([S1, S2, S3], 1)
        landmark = self.fc(landmark)
        # landmark = landmark.view(-1,68,2)
        pose = self.pose(x)
        return landmark, pose

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                
                # if m.bias is not None:
                #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #     m.weight.data.normal_(0, math.sqrt(2. / n))
                #     nn.init.xavier_normal_(m.weight.data)
                #     m.bias.data.fill_(0.02)
                # else:
                #     m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    net = PFLD_Net(0.25)
    x = torch.randn(2, 3, 112, 112)
    y = net(x)
    print(y)
    # net = net.to('cuda:2')
    # from torchsummary import summary
    # summary(net, (3, 112, 112))
