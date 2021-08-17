#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : config.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 16:56
'''

import os
import getpass
import torch
import numpy as np

class Config():
    def __init__(self, exp_name="h36m", input_n=10, output_n=10, dct_n=15, device="cuda:0", num_works=0, test_manner="all"):
        self.platform = getpass.getuser()
        assert exp_name in ["h36m", "cmu", "3dpw"]
        self.exp_name = exp_name

        self.p_dropout = 0.1
        self.train_batch_size = 16
        self.test_batch_size = 128
        self.lr = 2e-4
        self.lr_decay = 0.98
        self.n_epoch = 5000
        self.leaky_c = 0.2

        self.test_manner = test_manner
        self.input_n = input_n
        self.output_n = output_n
        self.seq_len = input_n + output_n
        self.dct_n = dct_n
        if self.output_n == 25:
            self.frame_ids = [1, 3, 7, 9, 13, 24]
        elif self.output_n == 10:
            self.frame_ids = [1, 3, 7, 9]

        if exp_name == "h36m":

            self.origin_noden = 32
            self.final_out_noden = 22

            self.dim_used_3d = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
            self.dim_repeat_22 = [9, 9, 14, 16, 19, 21]
            self.dim_repeat_32 = [16, 24, 20, 23, 28, 31]

            self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11], [12], [13], [14, 15, 16], [17], [18], [19, 20, 21]]
            self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
            self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

            self.I32_plot = np.array(
                [0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27,
                 28,
                 27, 30])
            self.J32_plot = np.array(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29,
                 30, 31])
            self.LR32_plot = np.array(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

            self.I22_plot = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
            self.J22_plot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
            self.LR22_plot = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

            self.I12_plot = np.array([4, 0, 4, 2, 4, 4, 6, 7, 4, 9, 10])
            self.J12_plot = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
            self.LR12_plot = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])

            self.I7_plot = np.array([2, 2, 2, 3, 2, 5])
            self.J7_plot = np.array([0, 1, 3, 4, 5, 6])
            self.LR7_plot = np.array([0, 1, 1, 1, 0, 0])

            self.I4_plot = np.array([0, 1])
            self.J4_plot = np.array([3, 2])
            self.LR4_plot = np.array([0, 1])
        elif exp_name == "cmu":

            self.origin_noden = 38
            self.final_out_noden = 25

            self.dim_used_3d = [3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 28, 30, 31, 32, 34, 35, 37]
            self.dim_repeat_22 = [9, 9, 9, 15, 15, 21, 21]
            self.dim_repeat_32 = [16, 20, 29, 24, 27, 33, 36]

            self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11, 12], [13], [14, 15], [16, 17, 18], [19], [20, 21], [22, 23, 24]]  # 其实是 Index2512, 为了保持统一没改名
            self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
            self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

            self.Index2510 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11, 12], [14, 15], [16, 17, 18], [20, 21],
                         [22, 23, 24]]
            self.Index105 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            self.Index53 = [[2], [0, 3], [1, 4]]

            self.I32_plot = np.array(
                [0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 16, 20, 21, 22, 23, 24, 25, 23, 27,
                 16, 29, 30, 31, 32, 33, 34, 32, 36])
            self.J32_plot = np.array(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30, 31, 32, 33, 34, 35, 36, 37])
            self.LR32_plot = np.array(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                 1, 1, 1])

            self.I22_plot = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 11, 9, 13, 14, 15, 16, 15, 9, 19, 20, 21, 22, 21])
            self.J22_plot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
            self.LR22_plot = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

            self.I12_plot = np.array([4, 0, 4, 2, 4, 4, 6, 7, 4, 9, 10])
            self.J12_plot = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
            self.LR12_plot = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1])

            self.I7_plot = np.array([2, 2, 2, 3, 2, 5])
            self.J7_plot = np.array([0, 1, 3, 4, 5, 6])
            self.LR7_plot = np.array([0, 1, 0, 0, 1, 1])

            self.I4_plot = np.array([0, 1])
            self.J4_plot = np.array([2, 3])
            self.LR4_plot = np.array([0, 1])

        self.device = device
        self.num_works = num_works
        self.ckpt_dir = os.path.join("./ckpt/", exp_name, "short_term" if self.output_n==10 else "long_term")
        if not os.path.exists(os.path.join(self.ckpt_dir, "models")):
            os.makedirs(os.path.join(self.ckpt_dir, "models"))
        if not os.path.exists(os.path.join(self.ckpt_dir, "images")):
            os.makedirs(os.path.join(self.ckpt_dir, "images"))

        if self.exp_name == "h36m":
            self.base_data_dir = os.path.join("F:\model_report_data\mocap_motion_prediction\data\human36mData3D\others", "h3.6m\dataset")
        elif self.exp_name == "cmu":
            self.base_data_dir = os.path.join("F:\model_report_data\mocap_motion_prediction", "data\cmu")



