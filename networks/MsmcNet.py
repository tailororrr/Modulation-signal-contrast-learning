# -*- coding: utf-8 -*- #
'''
    MsmcNet
'''

import torch.nn as nn
from torchsummary import summary

class MsmcNet_RML2016(nn.Module):

    def __init__(self, num_classes):
        super(MsmcNet_RML2016, self).__init__()

        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(5, 2), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.Block_2 = nn.Sequential(
            nn.Conv2d(30, 25, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_3 = nn.Sequential(
            nn.Conv2d(25, 15, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_4 = nn.Sequential(
            nn.Conv2d(15, 15, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(75, num_classes)
        self.out_features = 75
        # self.fc1 = nn.Linear(75, 128)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)

        x = self.flatten(x)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x

if __name__ == "__main__":
    ''' 测试网络结构构建是否构建正确，并打印每层参数 '''
    model = MsmcNet_RML2016(num_classes=11)
    model.cuda()
    print(model)
    # 统计网络参数及输出大小
    summary(model, (1, 128, 2), batch_size=1, device="cuda")
