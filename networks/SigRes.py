# 调制信号的resnet18/50
# 修改思路为，换掉第一个7*7的卷积，改为一个输出64通道，k=(5,1),s=(1,2),p=(2,1)

import torch
import torch.nn as nn
from torchsummary import summary

'''
BasicBlock：resnet-18和resnet-34的残差块结构；
Bottleneck：resnet-50，resnet-101和resnet-152的残差块结构；
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None):
        """定义BasicBlock残差块类
        
        参数：
            inplanes (int): 输入的Feature Map的通道数
            planes (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是planes*expansion
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# todo Bottleneck
class Bottleneck(nn.Module):
    """
    __init__
        in_channel：残差块输入通道数
        out_channel：残差块输出通道数
        stride：卷积步长
        downsample：在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
    """
    expansion = 4   # 残差块第3个卷积层的通道膨胀倍率
    def __init__(self, inplanes, planes, stride=(1,1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1,1), stride=(1,1), bias=False)   # H,W不变。C: in_channel -> out_channel
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3,1), stride=stride, bias=False, padding=(1,0))  # H/2，W/2。C不变
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=(1,1), stride=(1,1), bias=False)   # H,W不变。C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x    # 将原始输入暂存为shortcut的输出
        if self.downsample is not None:
            identity = self.downsample(x)   # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity     # 残差连接
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=24):
        super(ResNet, self).__init__()
        self.inplanes = 16  # 第一个残差块的输入通道数
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5,2), stride=(1,2), padding=(2,1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) #卷积到64*1，开始残差
        
        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 16, layers[0],stride=(1,1))
        self.layer2 = self._make_layer(block, 32, layers[1], stride=(2,1))
        self.layer3 = self._make_layer(block, 64, layers[2], stride=(2,1))
        self.layer4 = self._make_layer(block, 128, layers[3], stride=(2,1))

        # 这里需要测试，是使用自适应池化好还是使用两个全连接好
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP自适应平均池化
        self.maxpool2 = nn.MaxPool2d(kernel_size=(7, 1), stride=(7, 1)) #卷积到64*1，开始残差
        # self.fc = nn.Linear(256 * 32, num_classes)
        self.fc = nn.Linear(128, num_classes)
        self.out_features = 128
        # self.fc = nn.Linear(512 * 8, num_classes)  # 这个适用于9，18，34

        # self.fc = nn.Sequential(
        #     nn.Linear(512 * 8, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, num_classes)
        # )
    def _make_layer(self, block, planes, blocks, stride=1):
        """定义ResNet的一个Stage的结构
        
        参数：
            block (BasicBlock / Bottleneck): 残差块结构
            plane (int): 残差块中第一个卷积层的输出通道数
            bloacks (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """
        downsample = None
        # 在通道变换或者s变换的时候是需要下采样的
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),) # 下采样

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # 先添加一个block

        self.inplanes = planes * block.expansion # 输入的Feature Map的通道数
        # 
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes)) # 输出的通道数

        return nn.Sequential(*layers)

    # 从这里开始写，这样可以算
    def forward(self, x):

        x = self.conv1(x) # 第一个
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool2(x) #自适应平均池化
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet10_re(num_classes):
    model = ResNet(block = BasicBlock,layers = [1, 1, 1, 1], num_classes=11)
    return model

def resnet18_re():
    model = ResNet(block = BasicBlock,layers = [2, 2, 2, 2], num_classes=11)
    return model

def resnet34():
    model = ResNet(block = BasicBlock,layers = [3, 4, 6, 3], num_classes=24)
    return model

def resnet50():
    model = ResNet(block = Bottleneck,layers = [3, 4, 6, 3], num_classes=24)
    return model

if __name__ == "__main__":
    ''' 测试网络结构构建是否构建正确，并打印每层参数 '''
    #model = resnet18()
    model = resnet10_re(num_classes=11)
    model.cuda() 
    print(model)
    # 统计网络参数及输出大小 block, layers, 
    summary(model, (1, 128, 2), batch_size=1, device="cuda")