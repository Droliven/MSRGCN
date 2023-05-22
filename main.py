#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : main.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-25 16:29
'''

# ****************************************************************************************************************
# *********************************************** 环境部分 ********************************************************
# ****************************************************************************************************************

import numpy as np
import random
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def seed_torch(seed=3450):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

seed_torch()

# ****************************************************************************************************************
# *********************************************** 主体部分 ********************************************************
# ****************************************************************************************************************

import argparse
import pandas as pd
from pprint import pprint

from run import H36MRunner, CMURunner
from datas import define_actions, define_actions_cmu

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="cmu", help="h36m / cmu")
parser.add_argument('--input_n', type=int, default=10, help="")
parser.add_argument('--output_n', type=int, default=25, help="")
parser.add_argument('--dct_n', type=int, default=35, help="")
parser.add_argument('--device', type=str, default="cuda:0", help="")
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--test_manner', type=str, default="all", help="all / 8")
parser.add_argument('--debug_step', type=int, default=1, help="")
parser.add_argument('--is_train', type=bool, default='', help="")
parser.add_argument('--is_load', type=bool, default='', help="")
parser.add_argument('--model_path', type=str, default="", help="")

args = parser.parse_args()

print("\n================== Arguments =================")
pprint(vars(args), indent=4)
print("==========================================\n")

if args.exp_name == "h36m":
    r = H36MRunner(exp_name=args.exp_name, input_n=args.input_n, output_n=args.output_n, dct_n=args.dct_n, device=args.device, num_works=args.num_works,
                   test_manner=args.test_manner, debug_step=args.debug_step)
    acts = define_actions("all")

elif args.exp_name == "cmu":
    r = CMURunner(exp_name=args.exp_name, input_n=args.input_n, output_n=args.output_n, dct_n=args.dct_n,
                   device=args.device, num_works=args.num_works,
                   test_manner=args.test_manner, debug_step=args.debug_step)
    acts = define_actions_cmu("all")

if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()
else:
    errs = r.test()
    # errs = r.new_test_like_MotionMixerIJCAI22() # 注意要将 test_manner 改为 256，同时用预测结果25帧整体去计算损失并平均，（可能需要将 test_batchsize 改为256）

    col = r.cfg.frame_ids
    d = pd.DataFrame(errs, index=acts, columns=col)
    d.to_csv(f"{r.cfg.exp_name}_in{r.cfg.input_n}out{r.cfg.output_n}dctn{r.cfg.dct_n}_{r.cfg.test_manner}.csv", lineterminator="\n")

    # r.save(os.path.join(r.cfg.ckpt_dir, "models", '{}_in{}out{}dctn{}_best_epoch{}_err{:.4f}.pth'.format(r.cfg.exp_name, r.cfg.input_n, r.cfg.output_n, r.cfg.dct_n, r.start_epoch, np.mean(errs))),
    #        r.start_epoch, np.mean(errs), np.mean(errs))



