# 调制信号对比学习

使用对比学习进行调制信号特征表示，可以用的方法包括：simclr,moco,nnclr,swav,byol,simsiam

使用的数据增强方式包括：初等旋转变换，翻转，对称，时移等

使用的特征提取网络包括：四层卷积网络，改进的resnet网络（sig-res10）等

本仓库为论文《Achieving Efficient Feature Representation for Modulation Signal: A Cooperative Contrast Learning Approach》中对比实验的部分内容

## 代码使用
pkl2npy.py生成数据集

1 预训练

simclr.py 单一训练simclr的程序

self_train_main.py 多种对比学习方法的综合

2 微调

eval_semi_train.py 微调训练

eval_semi_test.py 微调测试（这里的测试是分开信噪比的结果）
