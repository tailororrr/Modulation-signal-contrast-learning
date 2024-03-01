# -*- coding: utf-8 -*- #
import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('TkAgg')

from RML2016all import loadNpy
from classes import modName

def showOriSignal(sample, label):
    ''' 绘制一个样本信号的图像 '''
    mod_name = str(modName[label], "utf-8")
    signal_data = sample[0]

    plt.figure(figsize=(6, 4))
    # plt.title(mod_name)
    # plt.xlabel('off')
    # plt.ylabel("off")

    plt.plot(signal_data[:, 0], c='tomato')
    plt.plot(signal_data[:, 1], c='c')
    # plt.legend(loc="upper right")

    #plt.show()
    plt.axis('off')
    plt.savefig("2.jpg", dpi=120)
    #plt.close()

def drawAllOriSignal(X, Y):
    """
    Args:
        X: numpy.ndarray(size = (bz, 1, 128, 2)), 可视化信号原始数据
        Y: numpy.ndarray(size = (bz,)), 可视化信号标签
    Returns:
        None
    Funcs:
        绘制所有信号输入样本的图像，并保存至相应标签的文件夹下
    """
    for idx in range(len(X)):
        if (idx+1)%50 == 0:
            print("{} complete!".format(idx+1))
        signal_data = X[idx][0]
        mod_name = str(modName[Y[idx]], "utf-8")

        plt.figure(figsize=(6, 4))
        # plt.title(mod_name)
        # plt.xlabel('N')
        # plt.ylabel("Value")

        plt.plot(signal_data[:, 0], c='red', linewidth=2.0, label = 'I')
        plt.plot(signal_data[:, 1], linewidth=2.0, label = 'Q')
        # plt.legend(loc="upper right")

        save_path = "fig/{}".format(mod_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.axis('off')
        plt.savefig(save_path + '/' + str(idx+1), dpi=120)
        plt.close()
        
    print(X.shape)
    print(Y.shape)
    print("Complete the drawing of all original signals !!!")

if __name__ == "__main__":
    train_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-test-2.npy'
    test_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-test-2.npy'
    x_train, y_train, x_test, y_test = loadNpy(train_path, test_path)
    print(x_train.shape, y_train.shape)
    drawAllOriSignal(X=x_train[:200], Y=y_train[:200])
    # for idx in range(len(x_train)):
    #     showOriSignal(x_train[], y_train[110])
