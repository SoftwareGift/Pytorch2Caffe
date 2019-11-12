import torch.nn as nn

# building blocks for mobilenet.
# name Conv and Conv_dw are following terms used in mobilenet paper (https://arxiv.org/pdf/1704.04861.pdf)

class Conv(nn.Module):
    """
    Conv block is convolutional layer followed by batch normalization and ReLU activation
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, use_relu6=False):
        super().__init__()
        self.layers = [
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channel)
        ]

        if use_relu6:
            self.layers.append(nn.ReLU6(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*self.layers)
        
    def forward(self, input):
        return self.model(input)
    

class Conv_dw_Conv(nn.Module):
    """
    Conv_dw is depthwise (dw) convolution layer followed by batch normalization and ReLU activation.
    Conv_dw_Conv is Conv_dw block followed by Conv block.
    Implemented Conv_dw_Conv instead of Conv_dw since in MobleNet, every Conv_dw is followed by Conv
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, use_relu6=False):
        super().__init__()
        self.layers = [
                nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, bias=False, groups=in_channel),
                nn.BatchNorm2d(in_channel)
        ]
        
        if use_relu6:
            self.layers.append(nn.ReLU6(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(Conv(in_channel, out_channel, kernel_size=1, stride=1, padding=0, use_relu6=use_relu6))

        self.model = nn.Sequential(*self.layers)
        
    def forward(self, input):
        return self.model(input)
    
    
# class MobileNetV1(nn.Module):
#     def __init__(self, num_classes=2, use_relu6=False):
#         super().__init__()
        
#         self.num_classes = num_classes
        
#         self.model = nn.Sequential(
#                 Conv(3, 32, stride=2, use_relu6=use_relu6),
#                 Conv_dw_Conv(32, 64, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(64, 128, kernel_size=3, stride=2, use_relu6=use_relu6),
#                 Conv_dw_Conv(128, 128, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(128, 256, kernel_size=3, stride=2, use_relu6=use_relu6),
#                 Conv_dw_Conv(256, 256, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(256, 512, kernel_size=3, stride=2, use_relu6=use_relu6),

#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),

#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=2, use_relu6=use_relu6),
#                 Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)
#         )
        
        
# #      self.avg_pool = nn.AvgPool2d(7)
#         self.avg_pool = nn.AvgPool2d(7)
#         self.fc = nn.Linear(512, num_classes)
        
#     def forward(self, input):
#         x = self.model(input)
# #         x = x.mean(3).mean(2)
#         x = self.avg_pool(x)
# #         x = self.flatten(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
# def mobilenet_v1():
#     return MobileNetV1(2)

class MobileNetV1(nn.Module):
    def __init__(self, num_classes, use_relu6=False):
        super().__init__()
        
        self.num_classes = num_classes
        
        
        self.block1 = Conv(3, 32, stride=2, use_relu6=use_relu6)
        self.block2 = Conv_dw_Conv(32, 64, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block3 = Conv_dw_Conv(64, 128, kernel_size=3, stride=2, use_relu6=use_relu6)
        self.block4 = Conv_dw_Conv(128, 128, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block5 = Conv_dw_Conv(128, 256, kernel_size=3, stride=2, use_relu6=use_relu6)
        self.block6 = Conv_dw_Conv(256, 256, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block7 = Conv_dw_Conv(256, 512, kernel_size=3, stride=2, use_relu6=use_relu6)

        self.block8 = Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block9 = Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block10 = Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block11 = Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)
        self.block12 = Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)

        self.block13 = Conv_dw_Conv(512, 512, kernel_size=3, stride=2, use_relu6=use_relu6)
        self.block14 = Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6)
        
        
        
        self.avg_pool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x) 
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

def mobilenet_v1():
    return MobileNetV1(2)