#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : short_term_main.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-28 19:20
'''

# ****************************************************************************************************************
# *********************************************** 环境部分 ********************************************************
# ****************************************************************************************************************

import numpy as np
import random
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
parser.add_argument('--output_n', type=int, default=10, help="")
parser.add_argument('--dct_n', type=int, default=15, help="")
parser.add_argument('--device', type=str, default="cuda:0", help="")
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--test_manner', type=str, default="8", help="all / 8")
parser.add_argument('--debug_step', type=int, default=1, help="")
parser.add_argument('--is_train', type=bool, default='', help="")
parser.add_argument('--is_load', type=bool, default='', help="")

# parser.add_argument('--model_path', type=str, default=os.path.join(r"E:\PythonWorkspace\MSRGCN\ckpt\pretrained", "h36m_in10out10dctn15_best_err36.3625.pth"), help="")

parser.add_argument('--model_path', type=str, default=os.path.join(r"E:\PythonWorkspace\MSRGCN\ckpt\pretrained", "cmu_in10out10dctn15_best_err24.8084.pth"), help="")

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

    col = r.cfg.frame_ids
    d = pd.DataFrame(errs, index=acts, columns=col)
    d.to_csv(f"{r.cfg.exp_name}_in{r.cfg.input_n}out{r.cfg.output_n}dctn{r.cfg.dct_n}_{r.cfg.test_manner}.csv", line_terminator="\n")

    # r.save(os.path.join(r.cfg.ckpt_dir, "models", '{}_in{}out{}dctn{}_best_epoch{}_err{:.4f}.pth'.format(r.cfg.exp_name, r.cfg.input_n, r.cfg.output_n, r.cfg.dct_n, r.start_epoch, np.mean(errs))),
    #        r.start_epoch, np.mean(errs), np.mean(errs))