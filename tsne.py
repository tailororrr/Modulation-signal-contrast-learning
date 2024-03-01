
import torch
import numpy as np
import torch.nn as nn
import argparse
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
from data.RML2016all import RMLDataset, loadNpy_self_perSNR
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

from scipy.optimize import linear_sum_assignment as linear_assignment
from networks.SigRes import resnet10_re

from selfsup.methods.simclr import SimCLR                    #1
from selfsup.methods.byol import BYOL                        #2
from selfsup.methods.simsiam import SimSiam                  #3
from selfsup.methods.dcl import DCL                          #5
from selfsup.methods.nnclr import NNCLR                     #6
from selfsup.methods.moco import MoCo                        #7
from selfsup.methods.swav import SwaV                       #7

def loadNpy_self_perSNR(train_path,snrs, process_IQ=False):
    x_train = []
    y_train = []
    train_dictionary = np.load(train_path, allow_pickle=True).item()
    for snr,mod_dict in train_dictionary.items():
        if snr == snrs: 
            print(snr)
            for mod,samples in mod_dict.items():
                for sample in samples:
                    x_train.append(sample)
                    # y_train.append(np.where(np.array(modName) == mod)[0][0]) 
                    y_train.append(np.where(np.array(modName) == mod)[0][0]) 

    # 为适应keras做的维度变换:(H, W, C), 但PyTorch为(N, C, H, W), 需更改维度
    x_train = np.asarray(x_train)[:,:,:,np.newaxis]    
    x_train = np.swapaxes(x_train, 1, 3)    # PyTorch
    # x_train = np.swapaxes(x_train, 2, 3)
    y_train = np.asarray(y_train)

    return x_train, y_train

# 通过对back进行推理
def inference(loader, simclr_model, device, args):
    feature_vector = []
    labels_vector = []
    e_methods = ['simsiam','nnclr']
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding

        with torch.no_grad():
            if args.method in e_methods:
                h, z, _ = simclr_model(x)
            else:
                h, z = simclr_model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

    
def get_inference(args,x_train,y_train):
    transform = transforms.Compose([ 
                                    # transforms.ToTensor()
                                    # waiting add
                                    ])
    # Train data
    train_dataset = RMLDataset(x_train, y_train, transform=transform)    # RML2016.10a数据集
    train_loader = DataLoader(train_dataset, \
                                    batch_size=110, \
                                    num_workers=2, \
                                    shuffle=True, \
                                    drop_last=False)

    # 导入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = MsmcNet_RML2016(num_classes = 11)
    encoder = resnet10_re(num_classes = 11)
    backbone = nn.Sequential(*list(encoder.children())[:-1])
    n_features = encoder.fc.in_features #获取fc的输入维度
    if args.method == 'simclr':
        model = SimCLR(backbone, n_features)
    elif args.method == 'byol':
        model = BYOL(backbone, n_features)
    elif args.method == 'simsiam':
        model = SimSiam(backbone, n_features)
    elif args.method == 'dcl':
        model = DCL(backbone, n_features)
    elif args.method == 'nnclr':
        model = NNCLR(backbone, n_features)
    elif args.method == 'moco':
        model = MoCo(backbone, n_features)
    elif args.method == 'swav':
        model = SwaV(backbone, n_features)        
        
    else:
        print('...')

    # 加载无监督预训练模型
    print(args.method)
    model_fp = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/selfsupervised/checkpoints/{}/resnet10_500_RML201610A_550_final.pth'.format(args.method)
    print(model_fp)
    model.load_state_dict(torch.load(model_fp))
    model = model.to(device)
    model.eval()

    # 特征提取
    print("### Creating features from pre-trained context model ###")
    train_x, train_y = inference(train_loader, model, device, args)
    print(len(train_x))
    print(train_x.shape)

    return train_x,train_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval_cluster")
    parser.add_argument("-method", type=str, default='simclr') 
    args = parser.parse_args()
    s = [6]

    train_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-test-2.npy'
    x_train, y_train = loadNpy_self_perSNR(train_path,snrs=6) #载入数据
    print(x_train.shape) 
    print(y_train.shape)
    print(y_train)

    x_train_cluster, y_train_label = get_inference(args, x_train, y_train)
    print(x_train_cluster.shape,y_train.shape) # (2200, 128) (2200,)

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(x_train_cluster)
    print(X_tsne.shape)

    font = {"color": "darkred",
            "size": 13, 
            "family" : "serif"}
    
    colors = np.array(["red","green","black","orange","purple","beige","cyan","magenta","pink","blue","gray"])
    fig, ax = plt.subplots()
    newcmp = LinearSegmentedColormap.from_list('chaos',colors)

    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train_label, alpha=0.9, s=30, cmap=newcmp)
    modName = ['GFSK','WBFM','AM-SSB','AM-DSB','QPSK','QAM16','CPFSK','BPSK','PAM4','QAM64','8PSK']
    legend = ax.legend(handles=scatter.legend_elements()[0], labels=modName,loc="lower left")

    plt.title("{}".format(args.method), fontsize=20)
    # plt.savefig('tsnefig/RML2016A/test_tsne_or.png', dpi=120)
    plt.savefig('test_tsne_{}_1.png'.format(args.method), dpi=120)