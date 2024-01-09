#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2024-01-09] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import argparse
import torch
import os
import onnx
import logging

from model.model import ModelArgs, Transformer


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='trace llama2')
    parser.add_argument('--input_length', type=int, default=1000,
                        help='length of tokens, 1000 for prefill, \
                              1 for decode')
    parser.add_argument('--cache_size', type=int, default=4096,
                        help='length of caches, 0 for prefill, \
                              others for decode')
    parser.add_argument('--n_layers', type=int, default=32,
                        help='number of layers')
    parser.add_argument('--group_conv', default=False, action='store_true',
                        help='use group_conv instead of nn.Linear')
    parser.add_argument('--block_size', type=int, default=32,
                        help='block_size used in group_conv')
    args = parser.parse_args()

    # prefill mode
    if args.input_length > 1:
        assert args.cache_size > args.input_length
        start_pos = 0
    # decode mode
    else:
        start_pos = args.cache_size - 1
    model_args = ModelArgs(n_layers=args.n_layers,
                           max_seq_len=args.cache_size,
                           group_conv=args.group_conv,
                           block_size=args.block_size,
                           start_pos=start_pos)
    model = Transformer(model_args)
    model.eval().cpu().float()

    logging.info(model)

    input_ids = torch.randint(0, model_args.vocab_size,
                              (1, args.input_length)).long().cpu()

    os.makedirs("exp", exist_ok=True)
    save_name = "exp/llama2-7B-chat-hf-{}-".format(
        "prefill" if args.input_length > 1 else "decode"
    ) + "inputlen{}-nlayers{}-cachesize{}".format(
            args.input_length, args.n_layers, args.cache_size)
    if args.group_conv:
        save_name += "-useGconv-block{}".format(args.block_size)
    save_name += ".onnx"

    # onnx_program = torch.onnx.dynamo_export(model, input_ids)
    # onnx_program.save(save_name)

    torch.onnx.export(
        model,
        input_ids,
        save_name,
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        input_names=['tokens'],
        output_names=['logits'],
        verbose=False
    )

    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
