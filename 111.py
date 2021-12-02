# -*- coding: utf-8 -*-
"""
# @Time    : 2021/11/18 14:46
# @Author  : Xinqi Chen
# @Email   : chenxq66@sjtu.edu.cn
# @File    : 111.py
"""
import numpy as np

data = np.load('./dataset/hhar/data_20_120.npy')
labels = np.load('./dataset/hhar/label_20_120.npy')
labels_1 = labels[:,:,0]
labels_2 = labels[:,:,1]
labels_3 = labels[:,:,2]
np.save('./dataset/hhar/act_label_20_120.npy', labels_3)
max_l1 = np.unravel_index(np.argmax(labels_1), labels_1.shape)
max_l2 = np.unravel_index(np.argmax(labels_2), labels_2.shape)
max_l3 = np.unravel_index(np.argmax(labels_3), labels_3.shape)
print(labels[max_l1[0], max_l1[1], 0])
print(labels[max_l2[0], max_l2[1], 1])
print(labels[max_l3[0], max_l3[1], 2])