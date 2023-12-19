#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-18] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import copy
import torch
import numpy as np


class HorizonGroupConv(torch.nn.Module):
    def __init__(self, module, block_size=32):
        super().__init__()
        original = copy.deepcopy(module)
        self.idim = module.weight.size(1)
        self.odim = module.weight.size(0)
        assert self.idim % block_size == 0
        assert module.bias is None  # llama2 不带 bias
        self.num_groups = self.idim // block_size

        self.group_conv = torch.nn.Conv1d(
            self.idim, self.odim * self.num_groups,
            kernel_size=1,
            groups=self.num_groups, bias=False)
        self.group_conv.weight = torch.nn.Parameter(
            module.weight.reshape(
                self.odim,
                self.num_groups,
                self.idim // self.num_groups,
                1
            ).transpose(0, 1).reshape(-1, self.idim // self.num_groups, 1)
        )

        self.check_equal(original)

    def forward(self, hidden):
        B, T, _ = hidden.size()
        hidden = hidden.transpose(1, 2)  # (B, Cin, D) -> (B, Cin, T)
        out = self.group_conv(hidden)  # (B, Cout * n_group, T)
        out = out.reshape(B, self.num_groups, self.odim, T)
        out = torch.sum(out, dim=1)  # (B, Cout, T)
        out = out.transpose(1, 2).contiguous()  # (B, Cout, T) -> (B, T, Cout)
        return out

    def check_equal(self, module):
        module.eval()
        self.eval()
        random_data = torch.randn(1, 8, self.idim)
        orig_result = module(random_data)
        new_result = self.forward(random_data)
        np.testing.assert_allclose(
            orig_result.detach().cpu().numpy(),
            new_result.detach().cpu().numpy(),
            rtol=1e-07, atol=1e-05)
        print("check HorizonGroupConv, pass!")


if __name__ == "__main__":
    linear = torch.nn.Linear(320, 640, bias=False)
    g_conv = HorizonGroupConv(linear, block_size=32)
