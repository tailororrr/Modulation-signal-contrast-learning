# -*- coding: utf-8 -*- #
import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

sys.path.append("../")
from configs import cfgs
from dataset.RML2016 import loadNpy
from dataset.RML2016_10a.classes import modName

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
        plt.title(mod_name)
        plt.xlabel('N')
        plt.ylabel("Value")

        plt.plot(signal_data[:, 0], label = 'I')
        plt.plot(signal_data[:, 1], color = 'red', label = 'Q')
        plt.legend(loc="upper right")

        save_path = "../figs/original_signal/{}".format(mod_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + str(idx+1))
        plt.close()
        
    print(X.shape)
    print(Y.shape)
    print("Complete the drawing of all original signals !!!")


def showOriSignal(sample, label):
    ''' 绘制一个样本信号的图像 '''
    mod_name = str(modName[label], "utf-8")
    signal_data = sample[0]

    plt.figure(figsize=(6, 4))
    plt.title(mod_name)
    plt.xlabel('N')
    plt.ylabel("Value")

    plt.plot(signal_data[:, 0], label = 'I')
    plt.plot(signal_data[:, 1], color = 'red', label = 'Q')
    plt.legend(loc="upper right")

    plt.show()
    # plt.savefig("1.jpg")
    plt.close()


def showImgSignal(sample, label):
    ''' 绘制一个信号样本的二维可视化图像 '''
    data = sample[0].T                      # 2*128
    data = data - np.min(data)
    data = data / np.max(data)
    mod_name = str(modName[label], "utf-8")
    # print(data.shape)
    h, sig_len = data.shape

    # 叠加信号，以便显示
    img_sig = np.empty([sig_len, sig_len], dtype = float)
    # for row in range(int(sig_len/h)):
    #     img_sig[row*h:row*h+h, :] = data
    for row in range(sig_len):
        if row<sig_len/2:
            img_sig[row:row+1, :] = data[0]
        else:
            img_sig[row:row+1, :] = data[1]
    img_sig = cv2.resize(img_sig, (sig_len*2,sig_len*2))
    cv2.imshow(mod_name, img_sig)
    cv2.waitKey(0)
    return img_sig


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = loadNpy(cfgs.train_path, cfgs.test_path)
    print(x_train.shape, y_train.shape)
    # drawAllOriSignal(X=x_train, Y=y_train)
    for idx in range(len(x_train)):
        showImgSignal(x_train[idx], y_train[idx])
        showOriSignal(x_train[idx], y_train[idx])

