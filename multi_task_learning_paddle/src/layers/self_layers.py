#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/20
# @Author  : dangjinming
# @File    : __init__.py

import paddle
from paddle.nn import  Linear, Layer
import numpy as np

class MTLLoss(Layer):
    r"""
    Multi-Task-Learning loss
    reference: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
    """
    def __init__(self, task_nums):
        super(MTLLoss, self).__init__()
        x = paddle.zeros([task_nums], dtype='float32')
        self.log_var2s = paddle.create_parameter(
            shape=x.shape,
            dtype=str(x.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(x))

    def forward(self, loss_list):
        loss = 0
        for i in range(len(self.log_var2s)):
            loss += loss_list[i] / paddle.exp(self.log_var2s[i]) + self.log_var2s[i] / 2
        return paddle.mean(loss)
