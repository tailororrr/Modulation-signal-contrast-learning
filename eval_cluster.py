# -*- coding: utf-8 -*- #
'''
----- 自监督的第二阶段，聚类的测试
backbone + kmeans
backbone ： 是利用无标签的数据预训练好的

在聚类测试中，分信噪比进行测试
'''

import torch
import numpy as np
import torch.nn as nn
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment

from data.RML2016all import RMLDataset, loadNpy_self_perSNR # 数据集
from networks.SigRes import resnet10_re
from networks.MsmcNet import MsmcNet_RML2016 # 网络模型
from selfsup.methods.simclr import SimCLR                    #1
from selfsup.methods.byol import BYOL                        #2
from selfsup.methods.simsiam import SimSiam                  #3
from selfsup.methods.dcl import DCL                          #5
from selfsup.methods.nnclr import NNCLR                     #6
from selfsup.methods.moco import MoCo                        #7
from selfsup.methods.swav import SwaV                       #7


# 通过对backbone进行推理
def inference(loader, model, device, args):
    feature_vector = []
    labels_vector = []
    e_methods = ['simsiam','nnclr']
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            if args.method in e_methods:
                h, z, _ = model(x)
            else:
                h, z = model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

# 对聚类的结果进行评估
def cluster_acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm

    # reassignment = dict(zip(row_ind, col_ind))
    accuracy = w[row_ind, col_ind].sum() / y_pred.size
    return accuracy

def eval_kmeans(args,x_train,y_train):
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
    print(n_features)
    if args.method == 'simclr':
        model = SimCLR(backbone, n_features)
    elif args.method == 'byol':
        model = BYOL(backbone)
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

    print(args.method)
    # 加载预训练模型
    model_fp = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/selfsupervised/checkpoints/simclr/resnet10_500_RML201610A_550_final.pth'
    model.load_state_dict(torch.load(model_fp))
    model = model.to(device)
    model.eval()

    # 特征提取
    print("### Creating features from pre-trained model ###")
    train_x, train_y = inference(train_loader, model, device, args)
    # print(len(train_x))
    # print(train_x.shape)

    # kmeans
    kmeans_model = KMeans(n_clusters=11, init="k-means++").fit(train_x)
    # print(len(kmeans_model.labels_))
    test_acc = cluster_acc(train_y, kmeans_model.labels_)
    print(f"Test ACC", test_acc)
    test_ari = adjusted_rand_score(train_y, kmeans_model.labels_)
    print(f"Test ARI", test_ari)
    test_nmi = normalized_mutual_info_score(train_y, kmeans_model.labels_)
    print(f"Test NMI", test_nmi)

    return test_acc, test_ari, test_nmi


if __name__ == "__main__":
    # 挂载参数
    parser = argparse.ArgumentParser(description="eval_cluster")
    parser.add_argument("-method", type=str, default='simclr') 
    args = parser.parse_args()
    train_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-test-2.npy' #测试的数据集

    # s = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    s = [6,10]
    # 对不同的信噪比进行评估
    for snr in s:
        max_acc = 0
        max_ari = 0
        max_nmi = 0
        x_train, y_train = loadNpy_self_perSNR(train_path,snr) #载入数据
        print(x_train.shape)
        # 设置测试的次数
        for i in range(2):
            acc, ari, nmi = eval_kmeans(args, x_train, y_train)
            if acc > max_acc:
                max_acc = acc
                max_ari = ari
                max_nmi = nmi

            with open('./log/cluster_results.txt' , 'a') as f:
                exp_setting = ' %s | %d ' %(args.method, snr)
                acc_str = 'best acc = %4.2f%%, best ari = %4.2f%%, best nmi = %4.2f%%' %(max_acc, max_ari, max_nmi)
                f.write( 'RML201610A | eval_cluster -- %s | %s \n' %(exp_setting,acc_str))

    with open('./log/cluster_results.txt' , 'a') as f:
        f.write('##'*30)