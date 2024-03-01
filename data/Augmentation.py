# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
信号的数据增强操作：
    1、Givremote：旋转变换
    2、Reversal_I、Reversal_Q： 翻转
    3、Centrosymmetric ： 中心对称
    4、Shiftsignal ： 移位
    5、Resizesignal ：放缩
    6、Gaussian_white ： 高斯白噪声 （效果差，不考虑使用）

    MultiViewDataInjector ：返回两个数据增强的数据

'''
import sys
import numpy as np
import random

sys.path.append("../")

# givens变换，初等旋转变换
class Givremote(object):
    """ 对输入的信号进行旋转操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            in_array = [np.pi/8, np.pi/2, np.pi*5/6, np.pi*2/3, np.pi/6, np.pi*3/5 , np.pi*5/8, np.pi*3/2 , np.pi]
            t = random.choice(in_array)
            c = np.cos(t) #cos0 = 1, cos(pi/2)= 0 ,cos(3pi/2)= 0
            s = np.sin(t) #sin0 = 0, sin(pi/2)= 1 ,sin(3pi/2)=-1
            I = c * sample[0, :, 0] + s * sample[0, :, 1]       # I'=cI-sQ Q'=sI+cQ 
            Q = (-s) * sample[0, :, 0] + c * sample[0, :, 1]
            sample_return[0, :, 0] = I.copy()
            sample_return[0, :, 1] = Q.copy()
            #sam = self.pil_to_tensor(sample)
            return sample_return
        return sample

class Reversal_I(object):
    """ 对输入的信号进行翻转操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob 
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            I = -sample[0, :, 0]
            sample_return[0, :, 0] = I.copy()
            return sample_return
        return sample

class Reversal_Q(object):
    """ 对输入的信号进行翻转操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            Q = -sample[0, :, 1]
            sample_return[0, :, 1] = Q.copy()
            return sample_return
        return sample

class Centrosymmetric(object):
    """ 对输入的信号进行中心对称操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            I = sample[0, :, 0][::-1]
            Q = sample[0, :, 1][::-1]
            sample_return[0, :, 0] = I.copy()
            sample_return[0, :, 1] = Q.copy()
            return sample_return
        return sample

'''
移位思想：
    例如{1， 2， 3， 4， 5， 6， 7，8}要向左平移2个单位
    先将前两个数据反转为{2，1}
    之后将后面的数据反转为{8，7，6，5，4，3}
    两个片段的相对位置没有改变得到了{2，1，8，7，6，5，4，3}
    最后将总数据进行反转便得到了{3，4，5，6，7，8，1，2}
'''

class Shiftsignal(object):
    """ 对输入的信号进行移位操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob
        self.t = 20 # 移位的个数

    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            I_n = list(reversed(list(reversed(sample[0, :, 0][:128-self.t]))+list(reversed(sample[0, :, 0][128-self.t:]))))
            Q_n = list(reversed(list(reversed(sample[0, :, 1][:128-self.t]))+list(reversed(sample[0, :, 1][128-self.t:]))))
            sample_return[0, :, 0] = I_n.copy()
            sample_return[0, :, 1] = Q_n.copy()
            return sample_return
        return sample

# 缩放0.8-1.2倍左右
class Resizesignal(object):
    """ 对输入的信号进行放缩操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob
        
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            s = random.randint(8,12)
            I = sample[0, :, 0] * s * 0.1
            Q = sample[0, :, 1] * s * 0.1
            sample_return[0, :, 0] = I.copy()
            sample_return[0, :, 1] = Q.copy()
            return sample_return
        return sample
        
# 高斯白噪声
# 白噪声：频率分布全频域，序列间互不相关。
# 高斯噪声：概率密度分布符合高斯分布。
# 高斯白噪声：它的幅度分布服从高斯分布，而它的功率谱密度又是均匀分布的

def wgn(x):
    # 修改添加的噪声范围，之前是哪个噪声就加什么噪声
    snr = np.random.randint(15,20)
    snr=10**(snr/10) # 这个跟信噪比有关
    xsum=0
    for d in x:
        xsum=xsum+abs(d)**2
    l=len(x)
    xpower = xsum / l
    npower = xpower / snr
    a=np.random.randn(l)*np.sqrt(npower)
    return a

class Gaussian_white(object):
    """ 对输入的信号加入高斯白噪声 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            I = wgn(sample[0, :, 0])
            Q = wgn(sample[0, :, 1])
            sample_return[0, :, 0] = I.copy()
            sample_return[0, :, 1] = Q.copy()
            return sample_return
        return sample
        

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):

        output = [transform(sample) for transform in self.transforms]
        return output


