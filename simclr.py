'''
simclr训练：
    训练数据使用不区分信噪比的全部数据 (176000, 1, 128, 2)
    注意数据增其方式
    损失函数：NTXentLoss
    设置 ProjectionHead(out_feature, 512, 128)

'''
import os
import argparse
import torch
from torch import nn
from torchvision import transforms

from networks.MsmcNet import MsmcNet_RML2016 # 网络模型
from networks.SigRes import resnet10_re

from data.RML2016all import RMLDataset, loadNpy_self
from data.Augmentation import Givremote, Reversal_I, Reversal_Q, Centrosymmetric, Shiftsignal, Resizesignal, MultiViewDataInjector

from selfsup.heads import SimCLRProjectionHead
from selfsup.ntx_ent_loss import NTXentLoss


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class SimCLR(nn.Module):
    def __init__(self, backbone, out_feature):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(out_feature, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x,z


def main(args):
    # 载入数据
    x_train, y_train  = loadNpy_self(args.train_path)                           
    print(x_train.shape)
    data_transforms = transforms.Compose([Givremote(0.5),
                                            Reversal_I(0.5),Reversal_Q(0.5),
                                            Centrosymmetric(0.5),
                                            Shiftsignal(0.5),
                                            Resizesignal(0.5)] ) # 数据增强
    train_dataset = RMLDataset( x_train,y_train,
                                transform = MultiViewDataInjector([data_transforms, data_transforms]) )                            
                                     
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, 
                                                num_workers = args.num_workers, 
                                                shuffle = True, drop_last = True)
 
    # backbone模型导入
    cnn = resnet10_re(num_classes = 11)
    # cnn = MsmcNet_RML2016(num_classes=11)
    out_feature = cnn.fc.in_features
    #print(cnn.out_features)
    backbone = nn.Sequential(*list(cnn.children())[:-1])  #  取消最后一个全连接
    # exit(0)
    print(backbone)
    model = SimCLR(backbone, out_feature)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    criterion = NTXentLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)

    print("Starting Training")
    for epoch in range(args.max_epochs):
        total_loss = 0
        for i, ((x0, x1), _)  in enumerate(dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            _, z0 = model(x0)
            _, z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch [%d/%d], iter %d, Loss: %.4f' %(epoch+1, args.max_epochs,  i+1, loss))
        avg_loss = total_loss / len(dataloader)
        
        print('Epoch [%d/%d], Val_Loss: %.4f' %(epoch+1, args.max_epochs, avg_loss))
        # 保存模型
        if (epoch+1) % 100 == 0 :
            checkpoint_name = '{}_{}_{}_{}.pth'.format(args.model, str(epoch+1), args.dataset_name,args.batch_size)
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/simclr', checkpoint_name))
        if scheduler:
            scheduler.step()  

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/simclr', checkpoint_name))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self_train_simclr")
    parser.add_argument("-method", type=str, default='simclr')        # 方法名称
    parser.add_argument("-dataset_name", type=str, default='RML201610A')
    parser.add_argument("-model", type=str, default='resnet10')
    # 数据集的路径，只有训练集
    parser.add_argument("-train_path", type=str, default='/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-train-8.npy')  
    parser.add_argument("-batch_size", type=int, default=550)
    parser.add_argument("-num_workers", type=int, default=2)
    parser.add_argument("-lr", type=float, default=0.06)
    parser.add_argument("-weight_decay", type=float, default=0.0004)
    parser.add_argument("-optimizer", type=str, default='SGD')
    parser.add_argument("-max_epochs", type=int, default=500)
    args = parser.parse_args()
    main(args)
