# -*- coding: utf-8 -*- #
'''
----- 自监督的第二阶段，分类器训练和评估---半监督的测试
backbone + classifier

backbone是训练好的, 使用少量数据对整个网络进行微调
   1、在eval_semi_train.py： 是利用少量带标签的数据进行微调，训练过程
   2、在eval_semi_test.py：是对不同信噪比下的数据进行测试，测试上一步微调后的模型

   （为了方便，在1中使用全部的信噪比数据进行小样本微调，在2中分信噪比进行测试）
'''

import torch
import numpy as np
import torch.nn as nn
import argparse
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.RML2016all import RMLDataset, loadNpy_self_perSNR
from eval_semi_train import Semi_classifier
from networks.SigRes import resnet10_re

from selfsup.methods.simclr import SimCLR                    #1
from selfsup.methods.byol import BYOL                        #2
from selfsup.methods.simsiam import SimSiam                  #3
from selfsup.methods.dcl import DCL                          #5
from selfsup.methods.nnclr import NNCLR                     #6
from selfsup.methods.moco import MoCo                        #7
from selfsup.methods.swav import SwaV                       #7
from utils.strategy import accuracy


def main(args,snr):

   x_test, y_test = loadNpy_self_perSNR(args.test_path, snrs=snr)

   transform = transforms.Compose([])
   test_dataset = RMLDataset(x_test, y_test, transform=transform)
   test_loader = DataLoader(test_dataset, \
                              batch_size=args.batch_size, \
                              num_workers=2, \
                              shuffle=True, \
                              drop_last=False)

   # 导入模型
   # 加载预训练模型
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # decoder = MsmcNet_RML2016(num_classes = 11)
   decoder = resnet10_re(num_classes = 11)
   backbone = nn.Sequential(*list(decoder.children())[:-1])
   # backbone = resnet10_re(num_classes = 11)
   n_features = decoder.fc.in_features

   if args.method == 'simclr':
      self_model = SimCLR(backbone, n_features)
   elif args.method == 'byol':
      self_model = BYOL(backbone, n_features)
   elif args.method == 'simsiam':
      self_model = SimSiam(backbone, n_features)
   elif args.method == 'dcl':
      self_model = DCL(backbone, n_features)
   elif args.method == 'nnclr':
      self_model = NNCLR(backbone, n_features)
   elif args.method == 'moco':
      self_model = MoCo(backbone, n_features)
   elif args.method == 'swav':
      self_model = SwaV(backbone, n_features)        
        
   else:
      print('...')

   n_classes = 11 
   model = Semi_classifier(self_model.backbone, n_features, n_classes)

   load_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/selfsupervised/checkpoints/{}/semi_RML201610A_{}_best.pth'.format(args.method, args.rate)

   model.load_state_dict(torch.load(load_path))
   model = model.to(device)

   sum = 0
   val_top1_sum = 0
   labels = []
   preds = []
   model.eval()
   for ims, label in test_loader:
      labels += label.numpy().tolist()
      input = Variable(ims).cuda()
      target = Variable(label).cuda()
      output = model(input)

      _, pred = output.topk(1, 1, True, True)
      preds += pred.t().cpu().numpy().tolist()[0]
      top1_val = accuracy(output.data, target.data, topk=(1,))
      print(top1_val)
      sum += 1
      val_top1_sum += top1_val[0]
   print(sum)
   avg_top1 = val_top1_sum / sum
   print(avg_top1)

   with open('./log/semi-results.txt' , 'a') as f:
      exp_setting = '%s | %s | %s ' %(args.method, snr, args.rate)
      acc_str = 'best acc = %4.2f%%' %(avg_top1)
      f.write( 'RML2016.04C -- %s | %s \n' %(exp_setting,acc_str)  )

if __name__ == "__main__":
   # 挂载参数
   parser = argparse.ArgumentParser(description="Semi_self")
   # 修改这个挂载的路径，导入方式
   parser.add_argument("-batch_size", type=int, default=110)        # 对不同的方法进行选择
   parser.add_argument("-max_epochs", type=int, default=500) 
   parser.add_argument("-lr", type=float, default=0.01) 
   parser.add_argument("-method", type=str, default='simclr') 
   parser.add_argument("-rate", type=float, default=0.01)
   parser.add_argument("-test_path", type=str, default='/media/hp3090/HDD-2T/WX/sup-RML-2016/datasets/RML201604C/all-db/all-test-2.npy')

   args = parser.parse_args()
   SNRs = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
   for snr in SNRs:
      main(args,snr)