# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        utils/strategy.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 项目中用到的一些工具函数杂项，包括学习率调整、准确率计算
                    等，需不断更新
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    <0> step_lr():
                        -- 学习率手动调整，依据轮次调整，与keras代码中保持一致
                    <1> accuracy():
                        -- 计算top k分类准确率
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/14 | 修改学习率调整策略，修复bug
# ------------------------------------------------------------------------
'''

def step_lr(epoch, lr):
    learning_rate = lr
    if epoch < 500:
        learning_rate = lr
    elif epoch % 500 == 0:
        learning_rate = lr * 0.5
    return learning_rate


def accuracy(output, target, topk=(1, 5)):
    """ 
    Funcs:
        计算top k分类准确率 
    Args:
        output: 一个batch的预测标签
        target：一个batch的真实标签
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
