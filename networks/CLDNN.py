# -*- coding: utf-8 -*- #
'''
    CLDNN
'''
import sys
import torch
import torch.nn as nn
from torchinfo import summary

class CLDNN(nn.Module):
    def __init__(self, num_classes):
        super(CLDNN, self).__init__()

        self.conv1d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(24,2))
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.lstm1 = nn.LSTM(32,128,batch_first=True)
        self.lstm2 = nn.LSTM(128,128,batch_first=True)
        self.conv1d_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(8,1), stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.conv1d_1(x)  #这里的输出是(N, )
        #x = self.relu(x)    
        #x = self.maxpool(x) #这里的输出是(N,64,60)
        x = torch.squeeze(x, dim=3)
        x = torch.transpose(x,1,2)
        x,_ = self.lstm1(x)
        #x = self.dropout(x)
        x,_ = self.lstm2(x)
        #x = self.dropout(x)
        x = torch.transpose(x,1,2)
        x = torch.unsqueeze(x,3)
        # print(x.shape)
        x = self.conv1d_2(x)
        x = self.maxpool2(x)
        x = self.flatten(x) #展平
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x) 
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    ''' 测试网络结构构建是否构建正确，并打印每层参数 '''
    model = CLDNN(num_classes=11)
    model.cuda()
    #print(torch.cuda.current_device())
    print(model)
    # 统计网络参数及输出大小
    summary(model, (1, 1, 128, 2), device="cuda")
    # summary(model, (1, 128, 2), batch_size=1, device="cuda")
