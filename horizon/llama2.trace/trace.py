#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-18] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import argparse
import torch
import os
import logging

from hbdk4.compiler.torch import statistics

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

    model_args = ModelArgs(n_layers=args.n_layers,
                           max_seq_len=args.cache_size,
                           group_conv=args.group_conv,
                           block_size=args.block_size)
    model = Transformer(model_args)
    model.eval().cpu().float()

    logging.info(model)

    input_ids = torch.randint(0, model_args.vocab_size,
                              (1, args.input_length)).long().cpu()
    # prefill mode
    if args.input_length > 1:
        assert args.cache_size > args.input_length
        start_pos = 0
    # decode mode
    else:
        start_pos = args.cache_size - 1
    dummy_inputs = (input_ids, torch.tensor(start_pos).long())
    traced_model = torch.jit.trace(model, dummy_inputs,
                                   check_trace=False, strict=False)

    os.makedirs("exp", exist_ok=True)
    save_name = "exp/llama2-7B-chat-hf-{}-".format(
        "prefill" if args.input_length > 1 else "decode"
    ) + "inputlen{}-nlayers{}-cachesize{}".format(
            args.input_length, args.n_layers, args.cache_size)
    if args.group_conv:
        save_name += "-useGconv-block{}".format(args.block_size)
    save_name += ".traced.pt"
    traced_model.save(save_name)
    logging.info("Trace Done")

    with open("{}.example_input".format(save_name), "w") as f:
        f.write("input_len = {}\n".format(args.input_length))
        f.write("start_pos = {}\n".format(start_pos))
        f.write("input_ids = torch.randint(0, 1000, (1, input_len)).long().cpu()\n")
        f.write("example_input = (input_ids, torch.tensor(start_pos).long())\n")

    traced_model(*(dummy_inputs))
    logging.info(traced_model.graph)
    statistics(traced_model, dummy_inputs)
    logging.info("Statistics Done")


if __name__ == "__main__":
    main()
