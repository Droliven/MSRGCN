#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : msrgcn.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 16:46
'''

import torch
import torch.nn as nn
from .layers import SingleLeftLinear, SingleRightLinear, PreGCN, GC_Block, PostGCN


class MSRGCN(nn.Module):
    def __init__(self, p_dropout, leaky_c=0.2, final_out_noden=22, input_feature=35):
        super(MSRGCN, self).__init__()
        # 左半部分
        self.first_enhance = PreGCN(input_feature=input_feature, hidden_feature=64, node_n=final_out_noden * 3,
                                    p_dropout=p_dropout, leaky_c=leaky_c)  # 35, 64, 66, 0.5
        self.first_left = nn.Sequential(
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),  # 64, 0.5, 66
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
        )

        self.first_down = nn.Sequential(
            SingleLeftLinear(input_feature=final_out_noden * 3, out_features=36, seq_len=64, p_dropout=p_dropout,
                             leaky_c=leaky_c),  # 66, 128, 64
        )

        self.second_enhance = PreGCN(input_feature=64, hidden_feature=128, node_n=36, p_dropout=p_dropout,
                                     leaky_c=leaky_c)
        self.second_left = nn.Sequential(
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
        )

        self.second_down = nn.Sequential(
            SingleLeftLinear(input_feature=36, out_features=21, seq_len=128, p_dropout=p_dropout, leaky_c=leaky_c),
            # 66, 36, 64
        )

        self.third_enhance = PreGCN(input_feature=128, hidden_feature=256, node_n=21, p_dropout=p_dropout,
                                    leaky_c=leaky_c)
        self.third_left = nn.Sequential(
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
        )

        self.third_down = nn.Sequential(
            SingleLeftLinear(input_feature=21, out_features=12, seq_len=256, p_dropout=p_dropout, leaky_c=leaky_c),
            # 66, 36, 64
        )

        self.fourth_enhance = PreGCN(input_feature=256, hidden_feature=512, node_n=12, p_dropout=p_dropout,
                                     leaky_c=leaky_c)
        self.fourth_left = nn.Sequential(
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),  # 64, 0.5, 66
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
        )

        # 右半部分
        self.fourth_right = nn.Sequential(
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
        )
        self.fourth_up = nn.Sequential(
            SingleLeftLinear(input_feature=12, out_features=21, seq_len=512, p_dropout=p_dropout, leaky_c=leaky_c),
            SingleRightLinear(input_feature=512, out_features=256, node_n=21, p_dropout=p_dropout, leaky_c=leaky_c),
        )

        self.third_right_crop = nn.Sequential(
            SingleLeftLinear(input_feature=42, out_features=21, seq_len=256, p_dropout=p_dropout, leaky_c=leaky_c),
        )
        self.third_right = nn.Sequential(
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
        )
        self.third_up = nn.Sequential(
            SingleLeftLinear(input_feature=21, out_features=36, seq_len=256, p_dropout=p_dropout, leaky_c=leaky_c),
            SingleRightLinear(input_feature=256, out_features=128, node_n=36, p_dropout=p_dropout, leaky_c=leaky_c)
        )

        self.second_right_crop = nn.Sequential(
            SingleLeftLinear(input_feature=72, out_features=36, seq_len=128, p_dropout=p_dropout, leaky_c=leaky_c),
        )
        self.second_right = nn.Sequential(
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
        )
        self.second_up = nn.Sequential(
            SingleLeftLinear(input_feature=36, out_features=final_out_noden * 3, seq_len=128, p_dropout=p_dropout,
                             leaky_c=leaky_c),
            SingleRightLinear(input_feature=128, out_features=64, node_n=final_out_noden * 3, p_dropout=p_dropout,
                              leaky_c=leaky_c)
        )

        self.first_right_crop = nn.Sequential(
            SingleLeftLinear(input_feature=final_out_noden * 3 * 2, out_features=final_out_noden * 3, seq_len=64,
                             p_dropout=p_dropout, leaky_c=leaky_c),
        )
        self.first_right = nn.Sequential(
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),  # 64, 0.5, 66
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
        )

        # 右边出口部分
        self.first_extra = nn.Sequential(
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
        )
        self.first_out = PostGCN(input_feature=64, hidden_feature=input_feature, node_n=final_out_noden * 3)

        self.second_extra = nn.Sequential(
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
        )
        self.second_out = PostGCN(input_feature=128, hidden_feature=input_feature, node_n=36)

        self.third_extra = nn.Sequential(
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
        )
        self.third_out = PostGCN(input_feature=256, hidden_feature=input_feature, node_n=21)

        self.fourth_extra = nn.Sequential(
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
        )
        self.fourth_out = PostGCN(input_feature=512, hidden_feature=input_feature, node_n=12)

    def forward(self, inputs):
        '''

        :param x: B, 66, 35
        :return:
        '''
        x_p22 = inputs['p22']
        x_p12 = inputs['p12']
        x_p7 = inputs['p7']
        x_p4 = inputs['p4']

        # 左半部分
        enhance_first_left = self.first_enhance(x_p22)  # B, 66, 64
        out_first_left = self.first_left(enhance_first_left) + enhance_first_left  # 残差连接
        second_left = self.first_down(out_first_left)  # 8, 36, 64

        enhance_second_left = self.second_enhance(second_left)  # 8, 36, 128
        out_second_left = self.second_left(enhance_second_left) + enhance_second_left  # 残差连接
        third_left = self.second_down(out_second_left)

        enhance_third_left = self.third_enhance(third_left)  # 8, 21, 256
        out_third_left = self.third_left(enhance_third_left) + enhance_third_left  # 残差连接
        fourth_left = self.third_down(out_third_left)

        enhance_bottom = self.fourth_enhance(fourth_left)  # 8, 12, 512
        bottom = self.fourth_left(enhance_bottom) + enhance_bottom  # 残差连接

        # 右半部分
        bottom_right = self.fourth_right(bottom) + bottom  # 残差连接

        in_third_right = self.fourth_up(bottom_right)
        cat_third = torch.cat((out_third_left, in_third_right), dim=-2)
        crop_third_right = self.third_right_crop(cat_third)
        third_right = self.third_right(crop_third_right) + crop_third_right  # 残差连接

        in_second_right = self.third_up(third_right)
        cat_second = torch.cat((out_second_left, in_second_right), dim=-2)
        crop_second_right = self.second_right_crop(cat_second)
        second_right = self.second_right(crop_second_right) + crop_second_right  # 残差连接

        in_first_right = self.second_up(second_right)
        cat_first = torch.cat((out_first_left, in_first_right), dim=-2)
        crop_first_right = self.first_right_crop(cat_first)
        first_right = self.first_right(crop_first_right) + crop_first_right  # 残差连接

        # 出口部分
        fusion_first = self.first_extra(first_right) + first_right  # 残差连接
        pred_first = self.first_out(fusion_first) + x_p22  # 大残差连接

        fusion_second = self.second_extra(second_right) + second_right  # 残差连接
        pred_second = self.second_out(fusion_second) + x_p12  # 大残差连接

        fusion_third = self.third_extra(third_right) + third_right  # 两重残差连接
        pred_third = self.third_out(fusion_third) + x_p7  # 大残差连接

        fusion_fourth = self.fourth_extra(bottom_right) + bottom_right  # 残差连接
        pred_fourth = self.fourth_out(fusion_fourth) + x_p4  # 大残差连接

        return {
            "p22": pred_first, "p12": pred_second, "p7": pred_third, "p4": pred_fourth
            # "out_p22": pred_first

        }


if __name__ == "__main__":
    m = MSRGCN(0.3).cuda()
    print(">>> total params: {:.2f}M\n".format(sum(p.numel() for p in m.parameters()) / 1000000.0))
    pass


