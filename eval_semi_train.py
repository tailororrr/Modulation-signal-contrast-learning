# -*- coding: utf-8 -*- #
'''
----- 自监督的第二阶段，分类器训练和评估---半监督的测试
backbone + classifier

backbone是训练好的, 使用少量数据对整个网络进行微调
   1、在eval_semi_train.py： 是利用少量带标签的数据进行微调，训练过程
   2、在eval_semi_test.py：是对不同信噪比下的数据进行测试，测试上一步微调后的模型

'''
import os
import time

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import argparse
from torchvision import transforms

from torch.utils.data import DataLoader
from networks.MsmcNet import MsmcNet_RML2016 # 网络模型
from networks.SigRes import resnet10_re

from data.RML2016all import RMLDataset, loadNpy_semi
from utils.strategy import step_lr, accuracy

from selfsup.methods.simclr import SimCLR                    #1
from selfsup.methods.byol import BYOL                        #2
from selfsup.methods.simsiam import SimSiam                  #3
from selfsup.methods.dcl import DCL                          #5
from selfsup.methods.nnclr import NNCLR                     #6
from selfsup.methods.moco import MoCo                        #7
from selfsup.methods.swav import SwaV                       #7

# Semi-classifier
class Semi_classifier(torch.nn.Module):
    def __init__(self, backbone, out_feature, num_classes):
        super(Semi_classifier, self).__init__()
        self.backbone = backbone
        self.bottleneck = nn.Sequential(
                nn.Linear(out_feature, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
            ) 
        self.linear = torch.nn.Linear(256, num_classes)
        
    def forward(self, x):
       x = self.backbone(x).flatten(start_dim=1)
       # print(x.shape)
       x = self.bottleneck(x)
       # print(x.shape)
       return self.linear(x)


def main(args):
   x_train, y_train, x_test, y_test = loadNpy_semi(args.train_path, args.test_path, args.rate, process_IQ=True)
   print(x_train.shape)
   print(y_train.shape)

   transform = transforms.Compose([ 
                                    # transforms.ToTensor()
                                    # waiting add
                                 ])
   # Train data
   train_dataset = RMLDataset(x_train, y_train, transform=transform)    # RML2016.10a数据集
   train_loader = DataLoader(train_dataset, \
                                 batch_size=args.batch_size, \
                                 num_workers=2, \
                                 shuffle=True, \
                                 drop_last=False)
   # Valid data
   valid_dataset = RMLDataset(x_test, y_test, transform=transform)
   test_loader = DataLoader(valid_dataset, \
                              batch_size=args.batch_size, \
                              num_workers=2, \
                              shuffle=True, \
                              drop_last=False)

   # 导入模型
   # 加载预训练模型
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # decoder = MsmcNet_RML2016(num_classes = 11)
   decoder = resnet10_re(num_classes = 11)
   out_feature = decoder.out_features
   backbone = nn.Sequential(*list(decoder.children())[:-1])

   # backbone = resnet10_re(num_classes = 11)
   if args.method == 'simclr':
      self_model = SimCLR(backbone,out_feature)
   elif args.method == 'byol':
      self_model = BYOL(backbone,out_feature)
   elif args.method == 'simsiam':
      self_model = SimSiam(backbone,out_feature)
   elif args.method == 'dcl':
      self_model = DCL(backbone,out_feature)
   elif args.method == 'nnclr':
      self_model = NNCLR(backbone,out_feature)
   elif args.method == 'moco':
      self_model = MoCo(backbone,out_feature)
   elif args.method == 'swav':
      self_model = SwaV(backbone,out_feature)        
        
   else:
      print('...')

   model_fp = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/selfsupervised/checkpoints/{}/resnet10_500_RML201610A_550_final.pth'.format(args.method)
   self_model.load_state_dict(torch.load(model_fp))

   n_classes = 11 
   model = Semi_classifier(self_model.backbone,out_feature,n_classes)
   model = model.to(device)

   criterion = nn.CrossEntropyLoss().cuda()  # 交叉熵损失

   sum = 0
   train_loss_sum = 0
   train_top1_sum = 0
   max_val_acc = 0
   lr = args.lr


   for epoch in range(args.max_epochs):

      ep_start = time.time()
      lr = step_lr(epoch, lr)
      # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

      model.train()
      top1_sum = 0
      for i, (signal, label) in enumerate(train_loader):
         input = Variable(signal).cuda()
         target = Variable(label).cuda().long()

         output = model(input)            # inference
         
         loss = criterion(output, target) # 计算交叉熵损失
         optimizer.zero_grad()
         loss.backward()                  # 反传
         optimizer.step()

         top1 = accuracy(output.data, target.data, topk=(1,))  # 计算top1分类准确率
         train_loss_sum += loss.data.cpu().numpy()
         train_top1_sum += top1[0]
         sum += 1
         top1_sum += top1[0]

         print('Epoch [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                  %(epoch+1, args.max_epochs, lr, train_loss_sum/sum, train_top1_sum/sum))

         sum = 0
         train_loss_sum = 0
         train_top1_sum = 0

      if epoch % 1 == 0 and epoch < args.max_epochs:
         val_loss, val_top1 = eval(model, test_loader, criterion)

         print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, max_val_acc: %4f'
                  %(epoch+1, args.max_epochs, val_loss, val_top1, max_val_acc))

         if val_top1[0].data > max_val_acc:
            max_val_acc = val_top1[0].data
            print('Taking snapshot...')
            torch.save(model.state_dict(), '{}/{}/semi_{}_{}_best.pth'.format('checkpoints', args.method,args.dataset, args.rate))  

   with open('./semi_eval_results.txt' , 'a') as f:
      exp_setting = '%s | %f' %(args.method,args.rate)
      acc_str = 'Test Acc = %4.2f' %(max_val_acc)
      f.write( 'simclr -- %s | %s \n' %(exp_setting,acc_str) )


    
# validation 测试部分
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1




if __name__ == "__main__":
   # 挂载参数
   parser = argparse.ArgumentParser(description="Semi_self")
   # 修改这个挂载的路径，导入方式
   parser.add_argument("-batch_size", type=int, default=110)        # 对不同的方法进行选择
   parser.add_argument("-max_epochs", type=int, default=500) 
   parser.add_argument("-lr", type=float, default=0.001) 
   parser.add_argument("-method", type=str, default='simclr') 
   parser.add_argument("-rate", type=float, default=0.01)           # 设置训练样本的数量,0.01,0.1,0.2,0.5
   parser.add_argument("-dataset", type=str, default='RML201610A')
   parser.add_argument("-train_path", type=str, default='/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-train-8.npy')
   parser.add_argument("-test_path", type=str, default= '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-test-2.npy')

   args = parser.parse_args()

   main(args)