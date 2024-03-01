'''
RML2016提出的论文使用的方法
'''

import sys
import torch
import torch.nn as nn
from torchsummary import summary


class CNN_RML2016(nn.Module):
    def __init__(self,num_classes):
        super(CNN_RML2016, self).__init__()

        self.model=nn.Sequential(
                nn.Conv2d(1,256,kernel_size=(3,1),padding=(2,0)),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256,80,kernel_size=(3,2),padding=(2,0)),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Flatten(),
                
                nn.Linear(10560,256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256,num_classes)
                )

    def forward(self, x):
        x = self.model(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x

if __name__ == "__main__":
    ''' 测试网络结构构建是否构建正确，并打印每层参数 '''
    model = CNN_RML2016(num_classes=11)
    model.cuda()
    print(model)
    # 统计网络参数及输出大小
    summary(model, (1, 128, 2), batch_size=1, device="cuda")
