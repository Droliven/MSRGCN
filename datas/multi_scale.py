#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : multi_scale.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 20:30
'''

import numpy as np

def p_down(mydata, Index):
    '''
    leng, features, seq_len
    '''
    leng, features, seq_len = mydata.shape
    mydata = mydata.reshape(leng, -1, 3, seq_len)  # x, 22, 3, 35

    da = np.zeros((leng, len(Index), 3, seq_len)) # x, 12, 3, 35
    for i in range(len(Index)):
        da[:, i, :, :] = np.mean(mydata[:, Index[i], :, :], axis=1)
    da = da.reshape(leng, -1, seq_len)
    return da

def downs_from_22(downs, down_key):

    for key1, key2, key3 in down_key:
        downs[key2] = p_down(downs[key1], key3)
    return downs

