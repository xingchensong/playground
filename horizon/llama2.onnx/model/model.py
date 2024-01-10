#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of
#   the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model.group_conv import HorizonGroupConv


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    # make SwiGLU hidden layer size multiple of large power of 2
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 4096
    group_conv: bool = False
    block_size: int = 32
    start_pos: int = 0


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for
                numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator
                for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :,
              None, :].expand(bs, slen, n_kv_heads, n_rep,
                              head_dim).reshape(bs, slen, n_kv_heads * n_rep,
                                                head_dim))


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear or HorizonGroupConv): Linear for queries.
            wk (nn.Linear or HorizonGroupConv): Linear for keys.
            wv (nn.Linear or HorizonGroupConv): Linear for values.
            wo (nn.Linear or HorizonGroupConv): Linear for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None \
            else args.n_kv_heads
        model_parallel_size = 1  # fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.start_pos = args.start_pos

        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        if args.group_conv:
            self.wq = HorizonGroupConv(self.wq, args.block_size)
            self.wk = HorizonGroupConv(self.wk, args.block_size)
            self.wv = HorizonGroupConv(self.wv, args.block_size)
            self.wo = HorizonGroupConv(self.wo, args.block_size)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            rope (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = xq * rope
        xk = xk * rope

        keys = torch.cat([cache_k, xk], dim=1)
        values = torch.cat([cache_v, xv], dim=1)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys,
                         self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values,
                           self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim)
        if mask is not None:
            # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores,
                              values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(
        self,
        args: ModelArgs,
    ):
        """
        Initialize the FeedForward module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            w1 (nn.Linear or HorizonGroupConv): Linear for the first layer.
            w2 (nn.Linear or HorizonGroupConv): Linear for the second layer.
            w3 (nn.Linear or HorizonGroupConv): Linear for the third layer.

        """
        super().__init__()
        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = torch.nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = torch.nn.Linear(args.dim, hidden_dim, bias=False)

        if args.group_conv:
            self.w1 = HorizonGroupConv(self.w1, args.block_size)
            self.w2 = HorizonGroupConv(self.w2, args.block_size)
            self.w3 = HorizonGroupConv(self.w3, args.block_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.start_pos = args.start_pos

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            rope (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        h = x + self.attention.forward(self.attention_norm(x),
                                       rope, mask, cache_k, cache_v)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (nn.Linear): Linear layer for final output.
            rope (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(params.dim,
                                      params.vocab_size,
                                      bias=False)
        if params.group_conv:
            self.output = HorizonGroupConv(self.output, params.block_size)

        self.start_pos = params.start_pos

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen),
                              float("-inf"),
                              device=tokens.device)
            mask = torch.triu(mask, diagonal=self.start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, rope, mask, cache_k, cache_v)
        h = self.norm(h)
        output = self.output(h).float()
        return output
