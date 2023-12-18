#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-18] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import argparse
import torch
import os

from hbdk4.compiler.torch import statistics

from model.model import ModelArgs, Transformer


def main():
    parser = argparse.ArgumentParser(description='trace llama2')
    parser.add_argument('--input_length', type=int, default=1000,
                        help='length of tokens, 1000 for prefill, \
                              1 for decode')
    parser.add_argument('--cache_size', type=int, default=4096,
                        help='length of caches, 0 for prefill, \
                              others for decode')
    parser.add_argument('--n_layers', type=int, default=32,
                        help='number of layers')
    args = parser.parse_args()

    model_args = ModelArgs(n_layers=args.n_layers,
                           max_seq_len=args.cache_size)
    model = Transformer(model_args)
    print(model)
    model.eval().cpu().float()
    input_ids = torch.randint(0, model_args.vocab_size,
                              (1, args.input_length)).long().cpu()
    dummy_inputs = (input_ids, torch.tensor(0).long())
    traced_model = torch.jit.trace(model, dummy_inputs,
                                   check_trace=False, strict=False)
    os.makedirs("exp", exist_ok=True)
    traced_model.save(
        "exp/llama2-7B-chat-hf-" +
        "inputlen{}-nlayers{}-cachesize{}.traced.pt".format(
            args.input_length, args.n_layers, args.cache_size))
    print("Trace Done")
    traced_model(*(dummy_inputs))
    print(traced_model.graph)
    statistics(traced_model, dummy_inputs)
    print("Statistics Done")


if __name__ == "__main__":
    main()
