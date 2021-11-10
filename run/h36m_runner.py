#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : h36m_runner.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 21:24
'''

from datas import H36MMotionDataset, get_dct_matrix, reverse_dct_torch, define_actions,draw_pic_gt_pred
from nets import MSRGCN, MSRGCNShortTerm
from configs.config import Config

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from pprint import pprint

def L2NormLoss_test(gt, out, frame_ids):  # (batch size,feature dim, seq len)
    '''
    gt: B, 66, 25
    '''
    t_3d = np.zeros(len(frame_ids))

    batch_size, features, seq_len = gt.shape
    gt = gt.permute(0, 2, 1).contiguous().view(batch_size, seq_len, -1, 3) # B, 25, 22, 3
    out = out.permute(0, 2, 1).contiguous().view(batch_size, seq_len, -1, 3) # B, 25, 22, 3
    for k in np.arange(0, len(frame_ids)):
        j = frame_ids[k]
        t_3d[k] = torch.mean(torch.norm(gt[:, j, :, :].contiguous().view(-1, 3) - out[:, j, :, :].contiguous().view(-1, 3), 2, 1)).cpu().data.numpy() * batch_size
    return t_3d

def L2NormLoss_train(gt, out):
    '''
    # (batch size,feature dim, seq len)
    等同于 mpjpe_error_p3d()
    '''
    batch_size, _, seq_len = gt.shape
    gt = gt.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
    out = out.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
    loss = torch.mean(torch.norm(gt - out, 2, dim=-1))
    return loss

def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class H36MRunner():
    def __init__(self, exp_name="h36m", input_n=10, output_n=10, dct_n=15, device="cuda:0", num_works=0, test_manner="all", debug_step=1):
        super(H36MRunner, self).__init__()

        # 参数
        self.start_epoch = 1
        self.best_accuracy = 1e15

        self.cfg = Config(exp_name=exp_name, input_n=input_n, output_n=output_n, dct_n=dct_n, device=device, num_works=num_works, test_manner=test_manner)
        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")
        with open(os.path.join(self.cfg.ckpt_dir, "config.txt"), 'w', encoding='utf-8') as f:
            f.write(str(self.cfg.__dict__))
        # 模型
        if self.cfg.output_n == 25:
            self.model = MSRGCN(self.cfg.p_dropout, self.cfg.leaky_c, self.cfg.final_out_noden, input_feature=self.cfg.dct_n)
        elif self.cfg.output_n == 10:
            self.model = MSRGCNShortTerm(self.cfg.p_dropout, self.cfg.leaky_c, self.cfg.final_out_noden, input_feature=self.cfg.dct_n)

        if self.cfg.device != "cpu":
            self.model.cuda(self.cfg.device)

        print(">>> total params: {:.2f}M\n".format(
            sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        self.lr = self.cfg.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 数据
        dct_m, i_dct_m = get_dct_matrix(self.cfg.seq_len)
        self.dct_m = torch.from_numpy(dct_m).float()
        self.i_dct_m = torch.from_numpy(i_dct_m).float()
        if self.cfg.device != "cpu":
            self.dct_m = self.dct_m.cuda(self.cfg.device, non_blocking=True)
            self.i_dct_m = self.i_dct_m.cuda(self.cfg.device, non_blocking=True)

        train_dataset = H36MMotionDataset(self.cfg.base_data_dir, actions="all", mode_name="train", input_n=self.cfg.input_n, output_n=self.cfg.output_n,
                                      dct_used=self.cfg.dct_n, split=0, sample_rate=2,
                                          down_key=[('p22', 'p12', self.cfg.Index2212),
                                              ('p12', 'p7', self.cfg.Index127),
                                              ('p7', 'p4', self.cfg.Index74)], test_manner=self.cfg.test_manner, global_max=0, global_min=0, device=self.cfg.device, debug_step=debug_step)
        print("train data shape {}".format(train_dataset.gt_all_scales['p32'].shape[0]))

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_works,
            pin_memory=True)

        self.global_max = train_dataset.global_max
        self.global_min = train_dataset.global_min

        self.test_loader = dict()
        for act in define_actions("all"):
            test_dataset = H36MMotionDataset(self.cfg.base_data_dir, actions=act, mode_name="test", input_n=self.cfg.input_n, output_n=self.cfg.output_n,
                                      dct_used=self.cfg.dct_n, split=1, sample_rate=2,
                                          down_key=[('p22', 'p12', self.cfg.Index2212),
                                              ('p12', 'p7', self.cfg.Index127),
                                              ('p7', 'p4', self.cfg.Index74)], test_manner=self.cfg.test_manner, global_max=self.global_max, global_min=self.global_min, device=self.cfg.device, debug_step=debug_step)

            self.test_loader[act] = DataLoader(
                dataset=test_dataset,
                batch_size=self.cfg.test_batch_size,
                shuffle=False,
                num_workers=self.cfg.num_works,
                pin_memory=True)
            print(">>> test {} data {}".format(act, test_dataset.gt_all_scales['p32'].shape[0]))

        self.summary = SummaryWriter(self.cfg.ckpt_dir)

    def save(self, checkpoint_path, best_err, curr_err):
        state = {
            "lr": self.lr,
            "best_err": best_err,
            "curr_err": curr_err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)


    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.cfg.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr = state["lr"]
        best_err = state['best_err']
        curr_err = state["curr_err"]
        print("load lr {}, curr_avg {}, best_avg {}.".format(state["lr"], curr_err, best_err))


    def train(self, epoch):
        self.model.train()
        average_loss = 0
        for i, (inputs, gts) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            b, cv, t_len = inputs[list(inputs.keys())[0]].shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * len(self.train_loader) + i + 1

            for k in inputs:
                inputs[k] = inputs[k].float().cuda(non_blocking=True, device=self.cfg.device)
                gts[k] = gts[k].float().cuda(non_blocking=True, device=self.cfg.device)

            outputs = self.model(inputs)

            losses = None
            for k in outputs:
                # 反 Norm
                outputs[k] = (outputs[k] + 1) / 2
                outputs[k] = outputs[k] * (self.global_max - self.global_min) + self.global_min

                # 回转空间
                outputs[k] = reverse_dct_torch(outputs[k], self.i_dct_m, self.cfg.seq_len)

                # loss
                loss_curr = L2NormLoss_train(gts[k], outputs[k])
                if losses is None:
                    losses = loss_curr
                else:
                    losses = losses + loss_curr
                self.summary.add_scalar(f"Loss/{k}", loss_curr, self.global_step)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            average_loss += losses.cpu().data.numpy()

        average_loss /= (i + 1)
        return average_loss

    def test(self, epoch=0):
        self.model.eval()

        frame_ids = self.cfg.frame_ids
        total_loss = np.zeros((len(define_actions("all")), len(frame_ids)))

        for act_idx, act in enumerate(define_actions("all")):
            count = 0

            for i, (inputs, gts) in enumerate(self.test_loader[act]):
                b, cv, t_len = inputs[list(inputs.keys())[0]].shape
                for k in inputs:
                    inputs[k] = inputs[k].float().cuda(non_blocking=True, device=self.cfg.device)
                    gts[k] = gts[k].float().cuda(non_blocking=True, device=self.cfg.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    # 反 Norm
                    for k in outputs:
                        outputs[k] = (outputs[k] + 1) / 2
                        outputs[k] = outputs[k] * (self.global_max - self.global_min) + self.global_min

                        # 回转空间
                        outputs[k] = reverse_dct_torch(outputs[k], self.i_dct_m, self.cfg.seq_len)

                    # 开始计算
                    mygt = gts['p32'].view(-1, self.cfg.origin_noden, 3, self.cfg.seq_len).clone()
                    myout = outputs['p22'].view(-1, self.cfg.final_out_noden, 3, self.cfg.seq_len)
                    mygt[:, self.cfg.dim_used_3d, :, :] = myout
                    mygt[:, self.cfg.dim_repeat_32, :, :] = myout[:, self.cfg.dim_repeat_22, :, :]
                    mygt = mygt.view(-1, self.cfg.origin_noden*3, self.cfg.seq_len)

                    loss = L2NormLoss_test(gts['p32'][:, :, self.cfg.input_n:], mygt[:, :, self.cfg.input_n:], self.cfg.frame_ids)
                    total_loss[act_idx] += loss
                    # count += 1
                    count += mygt.shape[0]
                    # ************ 画图
                    if act_idx == 0 and i == 0:
                        pred_seq = outputs['p22'].cpu().data.numpy()[0].reshape(self.cfg.final_out_noden, 3,
                                                                                self.cfg.seq_len)
                        gt_seq = gts['p22'].cpu().data.numpy()[0].reshape(self.cfg.final_out_noden, 3, self.cfg.seq_len)
                        for t in range(self.cfg.seq_len):
                            draw_pic_gt_pred(gt_seq[:, :, t], pred_seq[:, :, t], self.cfg.I22_plot, self.cfg.J22_plot,
                                             self.cfg.LR22_plot,
                                             os.path.join(self.cfg.ckpt_dir, "images", f"{epoch}_{act}_{t}.png"))

            total_loss[act_idx] /= count
            for fidx, frame in enumerate(frame_ids):
                self.summary.add_scalar(f"Test/{act}/{frame}", total_loss[act_idx][fidx], epoch)

        self.summary.add_scalar("Test/average", np.mean(total_loss), epoch)
        for fidx, frame in enumerate(frame_ids):
            self.summary.add_scalar(f"Test/avg{frame}", np.mean(total_loss[:, fidx]), epoch)
        return total_loss


    def run(self):
        for epoch in range(self.start_epoch, self.cfg.n_epoch + 1):

            if epoch % 2 == 0:
                self.lr = lr_decay(self.optimizer, self.lr, self.cfg.lr_decay)
            self.summary.add_scalar("LR", self.lr, epoch)

            average_train_loss = self.train(epoch)

            if average_train_loss < self.best_accuracy:
                self.best_accuracy = average_train_loss
                self.save(os.path.join(self.cfg.ckpt_dir, "models",
                                 '{}_in{}out{}dctn{}_best_epoch{}_err{:.4f}.pth'.format(self.cfg.exp_name,
                                                                                        self.cfg.input_n,
                                                                                        self.cfg.output_n,
                                                                                        self.cfg.dct_n, epoch,
                                                                                        average_train_loss)),
                    self.best_accuracy, average_train_loss)

            self.save(os.path.join(self.cfg.ckpt_dir, "models",
                                   '{}_in{}out{}dctn{}_last.pth'.format(self.cfg.exp_name, self.cfg.input_n,
                                                                        self.cfg.output_n, self.cfg.dct_n)),
                      self.best_accuracy, average_train_loss)

            if epoch % 1 == 0:
                loss_l2_test = self.test(epoch)

                print('Epoch: {},  LR: {}, Current err test avg: {}'.format(epoch, self.lr, np.mean(loss_l2_test)))


if __name__ == '__main__':
    pass