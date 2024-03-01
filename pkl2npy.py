# -*- coding: UTF8 -*-
'''
将.pkl数据格式转换为.npy格式的，保存为字典的形式，相当于哈希表；
划分训练测试 8/2
--------------------------------------------------------------------------
    对文件中的数据简单整理分类，形成下列字典：
    {
        SNR1：{
        "调制方式1":DATA[samplesNum, 2, 128];
        "调制方式2":DATA[samplesNum, 2, 128];
        ...;
        }
        SNR2：{
        "调制方式1":DATA[samplesNum, 2, 128];
        "调制方式2":DATA[samplesNum, 2, 128];
        ...;
        }
        ...
    }
'''    
import pickle
import numpy as np
import random

# path = '2016.04C.multisnr.pkl'
path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016.10a_dict.pkl'       # 修改文件路径
f = open(path,'rb')
f_data = pickle.load(f, encoding='bytes')

train_set = {}
test_set = {}

train_path = "RML2016_10A/all-train-8.npy"                                      # 修改训练集保存路径
test_path = "RML2016_10A/all-test-2.npy"                                        # 修改测试集保存路径

for name,samples in f_data.items():
    MOD,SNR = name[0],name[1]   # name: (b'QPSK', -8) # (调制方式，信噪比)      
    if SNR not in train_set:
        train_set[SNR] = {}
        test_set[SNR] = {}

    samplesNum = len(samples)
    train_idx = random.sample(range(samplesNum), int(samplesNum*0.8))
    test_idx = []
    for i in range(samplesNum):
        if i not in train_idx:
            test_idx.append(i)

    train_set[SNR][MOD] = samples[train_idx, :, :]
    test_set[SNR][MOD] = samples[test_idx, :, :]

# 保存训练集、测试集至.npy
print(len(train_set))
print(train_set.keys())
print(train_set[2])
np.save(train_path, train_set)
print("------------------> Have saved train file:" + train_path)

print(len(test_set))
np.save(test_path, test_set)
print("------------------> Have saved test file:" + test_path)
print("\n\nFinished all files!!!\n\n")

