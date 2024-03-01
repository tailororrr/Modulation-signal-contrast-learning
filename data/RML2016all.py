# -*- coding: utf-8 -*- #
'''

RML2016数据集的读取：
    1、npy文件读取，包含数据预处理
        训练/测试正常读取，半监督数据读取，自监督预训练数据读取，信噪比选择读取
    2、定义RMLDataset类，返回(N, C, H, W)维度的数据

'''

import numpy as np
from torch.utils.data.dataset import Dataset

# sys.path.append("../")

modName = [
    b'GFSK', b'WBFM', b'AM-SSB', b'AM-DSB', b'QPSK', b'QAM16',
    b'CPFSK', b'BPSK', b'PAM4', b'QAM64', b'8PSK'
 ]

def processIQ(x_train):
    ''' 对两路信号分别进行预处理，结合两路为复数，除以标准差，再分离实部虚部到两路 '''
    for sample in x_train:
        sample_complex = sample[0, :, 0] + sample[0, :, 1] * 1j
        sample_complex = sample_complex / np.std(sample_complex)
        sample[0, :, 0] = sample_complex.real
        sample[0, :, 1] = sample_complex.imag
    return x_train

class RMLDataset(Dataset):
    ''' 定义RMLDataset类，继承Dataset方法，并重写__getitem__()和__len__()方法 '''
    def __init__(self, data_root, data_label, transform):
        ''' 初始化函数，得到数据 '''
        self.data = data_root
        self.label = data_label
        self.transform = transform

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回 '''
        data = self.data[index]
        labels = self.label[index]
        data = self.transform(data)
        return data, labels

    def __len__(self):
        ''' 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼 '''
        return len(self.data)


# 正常训练时候的数据载入
def loadNpy(train_path, test_path, process_IQ=True):
    x_train = []
    y_train = []
    x_test = [] 
    y_test = []
    train_dictionary = np.load(train_path, allow_pickle=True).item()
    test_dictionary = np.load(test_path, allow_pickle=True).item()
    for snr,mod_dict in train_dictionary.items():
        #if snr >= 0:
        for mod,samples in mod_dict.items():
            for sample in samples:
                x_train.append(sample)
                y_train.append(np.where(np.array(modName) == mod)[0][0]) 
    for snr,mod_dict in test_dictionary.items():   
        # if snr >=0 :         
        for mod,samples in mod_dict.items():
            for sample in samples:
                x_test.append(sample)
                y_test.append(np.where(np.array(modName) == mod)[0][0])

    # PyTorch为(N, C, H, W), 需更改维度
    x_train = np.asarray(x_train)[:,:,:,np.newaxis]    
    x_train = np.swapaxes(x_train, 1, 3)   

    x_test = np.asarray(x_test)[:,:,:,np.newaxis]
    x_test = np.swapaxes(x_test, 1, 3)  
       
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # 训练数据随机打乱
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    x_train = x_train[index,:,:,:]
    y_train = y_train[index]
    
    # IQ数据预处理，分别除以方差进行归一化
    if process_IQ:
        x_train = processIQ(x_train) 
        x_test = processIQ(x_test)
    return x_train, y_train, x_test, y_test


# 半监督训练的数据载入，多加参数rate=0.1，训练数据的比例
def loadNpy_semi(train_path, test_path, rate, process_IQ=True):
    x_train = []
    y_train = []
    x_test = [] 
    y_test = []
    train_dictionary = np.load(train_path, allow_pickle=True).item()
    test_dictionary = np.load(test_path, allow_pickle=True).item()

    # 训练数据读取，不同的比例
    for snr,mod_dict in train_dictionary.items():
        if snr >=0 :
            for mod,samples in mod_dict.items():   
                for sample in samples[:int(len(samples)*rate+0.49)]:
                    x_train.append(sample)
                    y_train.append(np.where(np.array(modName) == mod)[0][0]) 
    # 测试数据读取，保持不变
    for snr,mod_dict in test_dictionary.items():   
        if snr >=0 :         
            for mod,samples in mod_dict.items():
                for sample in samples:                                                                                                                
                    x_test.append(sample)
                    y_test.append(np.where(np.array(modName) == mod)[0][0])

    x_train = np.asarray(x_train)[:,:,:,np.newaxis]    
    x_train = np.swapaxes(x_train, 1, 3)    # PyTorch
    x_test = np.asarray(x_test)[:,:,:,np.newaxis]
    x_test = np.swapaxes(x_test, 1, 3)      # PyTorch
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # 训练数据随机打乱
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    x_train = x_train[index,:,:,:]
    y_train = y_train[index]
    
    # IQ两路预处理
    if process_IQ:
        x_train = processIQ(x_train) 
        x_test = processIQ(x_test)
    return x_train, y_train, x_test, y_test


# 自监督训练的数据载入，只输入训练数据
def loadNpy_self(train_path, process_IQ=False):
    x_train = []
    y_train = []
    train_dictionary = np.load(train_path, allow_pickle=True).item()
    for snr,mod_dict in train_dictionary.items():
        #if snr >= 0: 
        for mod,samples in mod_dict.items():
            for sample in samples:
                x_train.append(sample)
                y_train.append(np.where(np.array(modName) == mod)[0][0]) 

    x_train = np.asarray(x_train)[:,:,:,np.newaxis]    
    x_train = np.swapaxes(x_train, 1, 3)    
    y_train = np.asarray(y_train)

    # 训练数据随机打乱
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    x_train = x_train[index,:,:,:]
    y_train = y_train[index]
    
    # IQ两路预处理
    if process_IQ:
        x_train = processIQ(x_train) 
    return x_train, y_train

# 自监督训练的数据载入
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
                    y_train.append(np.where(np.array(modName) == mod)[0][0]) 

    # 为适应keras做的维度变换:(H, W, C), 但PyTorch为(N, C, H, W), 需更改维度
    x_train = np.asarray(x_train)[:,:,:,np.newaxis]    
    x_train = np.swapaxes(x_train, 1, 3)    # PyTorch
    # x_train = np.swapaxes(x_train, 2, 3)
    y_train = np.asarray(y_train)
    if process_IQ:
        x_train = processIQ(x_train) 
    return x_train, y_train

    # return x_train, y_train


# # 单信噪比的数据载入
# def loadNpy_self_persnr(train_path, snrs=0, process_IQ=False):
#     x_train = []
#     y_train = []
#     train_dictionary = np.load(train_path, allow_pickle=True).item()
#     s = [[6],[8],[10],[6,8,10]]
#     snrs_1 = s[snrs]
#     print(snrs_1)
#     for snr,mod_dict in train_dictionary.items():
#         if snr in snrs_1: 
#             for mod,samples in mod_dict.items():
#                 for sample in samples:
#                     x_train.append(sample)
#                     y_train.append(np.where(np.array(modName) == mod)[0][0]) 

#     # 为适应keras做的维度变换:(H, W, C), 但PyTorch为(N, C, H, W), 需更改维度
#     x_train = np.asarray(x_train)[:,:,:,np.newaxis]    
#     x_train = np.swapaxes(x_train, 1, 3)    # PyTorch
#     # x_train = np.swapaxes(x_train, 2, 3)
#     y_train = np.asarray(y_train)

#     # 训练数据随机打乱
#     index = np.arange(len(y_train))
#     np.random.shuffle(index)
#     x_train = x_train[index,:,:,:]
#     y_train = y_train[index]
    
#     # IQ两路预处理
#     if process_IQ:
#         x_train = processIQ(x_train) 
#     return x_train, y_train


if __name__ == "__main__":
    ''' 测试dataLoader是否正常读取、处理数据 '''

    train_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-train-8.npy'
    test_path = '/media/hp3090/HDD-2T/WX/RMLsig_ALL/datasets/RML2016_10A/all-test-2.npy'

    x_train, y_train, x_test, y_test = loadNpy(train_path, test_path)
    x_train, y_train, x_test, y_test = loadNpy_semi(train_path, test_path, rate=0.1)

    print(x_train.shape) # (176000, 1, 128, 2) :20*11*800
    print(y_train.shape) 
    print(x_test.shape) # (44000, 1, 128, 2)
    print(y_test.shape)
    # transform = transforms.Compose([ transforms.ToTensor()
    #                                 # waiting add
    #                                 ])
    # # 通过RMLDataset将数据进行加载，返回Dataset对象，包含data和labels
    # torch_data = RMLDataset(x_train, y_train, transform=transform)
    # # 通过DataLoader读取数据
    # datas = DataLoader( torch_data, \
    #                     batch_size=128, \
    #                     num_workers=2, \
    #                     shuffle=True, \
    #                     drop_last=False)
    # for i, data in enumerate(datas):
    #     # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    #     print("第 {} 个Batch \n{}".format(i, data))
    #     print("Size：", len(data[0]))
    
    # print(x_train.shape) # (3597, 1, 2, 1024)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
