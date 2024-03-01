
'''

# 写一个可以跑所有方法的主文件，用来训练第一阶段，做好不同方法的引用设计
# 可以选择的方法 simclr，byol，simsiam，dcl,nnclr，moco, swav

修改自：https://github.com/lightly-ai/lightly

'''

import torch
from torch import nn

import os
import argparse

from networks.MsmcNet import MsmcNet_RML2016 # 网络模型
from networks.SigRes import resnet10_re
from torchvision import transforms
from data.RML2016all import RMLDataset, loadNpy_self

from data.Augmentation import MultiViewDataInjector
from data.Augmentation import Givremote, Reversal_I, Reversal_Q, Centrosymmetric, Shiftsignal, Resizesignal, MultiViewDataInjector

# 不同方法的导入
# 可以选择的方法 simclr，byol，simsiam，dcl,nnclr，moco, swav
from selfsup.methods.simclr import Simclr_train                    #1
from selfsup.methods.byol import Byol_train                        #2
from selfsup.methods.simsiam import Simsiam_train                  #3
from selfsup.methods.dcl import Dcl_train                          #4
from selfsup.methods.nnclr import Nnclr_train                      #5
from selfsup.methods.moco import Moco_train                        #6
from selfsup.methods.swav import Swav_train                        #7

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(args):

    # 导入数据集
    x_train, y_train  = loadNpy_self(args.train_path) #载入数据
    print(x_train.shape)
    # 数据增强
    data_transforms = transforms.Compose([
                                            Givremote(0.5),
                                            Reversal_I(0.5),Reversal_Q(0.5),
                                            Centrosymmetric(0.5),
                                            Shiftsignal(0.5),
                                            Resizesignal(0.5)] ) 

    train_dataset = RMLDataset(
        x_train, y_train,
        transform = MultiViewDataInjector([data_transforms, data_transforms]) )             
                                
    dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size = args.batch_size, 
                                                num_workers = args.num_workers, 
                                                shuffle = True, 
                                                drop_last = True )
 
    # backbone模型导入
    cnn = resnet10_re(num_classes = 11)
    # cnn = MsmcNet_RML2016(num_classes=11)
    out_feature = cnn.fc.in_features
    backbone = nn.Sequential(*list(cnn.children())[:-1])
    # print(backbone)

    print('method:',args.method)
    if args.method == 'simclr':
        Simclr_train(args,backbone,out_feature,dataloader)

    elif args.method == 'byol':
        Byol_train(args,backbone,out_feature,dataloader)

    elif args.method == 'simsiam':
        Simsiam_train(args,backbone,out_feature,dataloader)

    elif args.method == 'dcl':
        Dcl_train(args,backbone,out_feature,dataloader)

    elif args.method == 'nnclr':
        Nnclr_train(args,backbone,out_feature,dataloader)

    elif args.method == 'moco':
        Moco_train(args,backbone,out_feature,dataloader)

    elif args.method == 'swav':
        Swav_train(args,backbone,out_feature,dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self_train_main")

    parser.add_argument("-method", type=str, default='simclr')        # 对不同的方法进行选择
    parser.add_argument("-dataset_name", type=str, default='RML201610A')
    parser.add_argument("-model", type=str, default='resnet10')
    parser.add_argument("-train_path", type=str, default='/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-train-8.npy')
    parser.add_argument("-batch_size", type=int, default=550)
    parser.add_argument("-num_workers", type=int, default=2)
    parser.add_argument("-lr", type=float, default=0.06)
    parser.add_argument("-weight_decay", type=float, default=0.0004)
    parser.add_argument("-optimizer", type=str, default='SGD')
    parser.add_argument("-max_epochs", type=int, default=2)
    args = parser.parse_args()
    main(args)
