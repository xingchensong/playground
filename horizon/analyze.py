#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-07] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import json
import sys


profile_path = sys.argv[1]

with open(profile_path, "r") as f:
    cont = f.readlines()
    tot_result = json.loads("".join(cont[:13]))
    result = json.loads("".join(cont[14:]))

    print("FPS/latency: {:.3f}/{:.3f}ms".format(
        round(tot_result['perf_result']['FPS'], 2),
        round(tot_result['perf_result']['average_latency'], 2)
    ))
    print("BPU + CPU: {:.3f}ms + {:.3f}ms".format(
        round(result["processor_latency"]["BPU_inference_time_cost"]["avg_time"], 2),
        round(result["processor_latency"]["CPU_inference_time_cost"]["avg_time"], 2)
    ))
    num_subgraph = 0
    latency = result["model_latency"]

    dic = {}
    for (k, v) in latency.items():
        if "BPU_torch_jit_subgraph" in k:
            num_subgraph += 1
        if "layout_convert" in k and "torch_jit_subgraph" in k:
            name = "input_layout_convert" if "input_layout_convert" in k \
                else "output_layout_convert"
        else:
            name = k.split('_')[0]
        if name not in dic:
            dic[name] = v["avg_time"]
        else:
            dic[name] = dic[name] + v["avg_time"]
    sorted_dic = dict(sorted(dic.items(), key=lambda kv: -kv[1]))
    print("NumSubGraph :{}".format(num_subgraph))
    print("==========Sorted BPU/CPU (sum)OPs===============")
    for (k, v) in sorted_dic.items():
        print("{}: {:.3f}".format(k, v))

    topk = 15
    print("==========Sorted CPU (standalone)OPs, Top {}=========".format(topk))
    dic.clear()
    for (k, v) in latency.items():
        if k.split('_')[0] == "BPU":
            continue
        name = k
        if name not in dic:
            dic[name] = v["avg_time"]
