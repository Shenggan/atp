import math

import torch
import torch.nn as nn
from spmd import Shard, DeviceMesh

from atp.layer import ATPLinear
from atp.distributed import get_default_mesh


class AttentionCore(nn.Module):

    def __init__(self,
                 attention_head_size: int,
                 attention_dropout: float) -> None:
        super().__init__()
        self.attention_head_size = attention_head_size
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.softmax = nn.Softmax(dim=-1)

        # (avoid the const tensor init when forward)
        self.causal_mask = None
        self.where_const = -1e4

    def forward(self, q, k, v, attention_mask):
        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)

        # (avoid the const tensor init when forward)
        if self.causal_mask is None:
            q_len, k_len = q.size(-2), k.size(-2)
            self.causal_mask = torch.tril(
                torch.ones((q_len, k_len), dtype=torch.uint8,
                           device="cuda")).view(1, 1, q_len, k_len).bool()
        x = torch.where(self.causal_mask, x, self.where_const)
        if attention_mask is not None:
            x = x + attention_mask
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (-1,)
        x = x.reshape(new_context_layer_shape)

        return x


class ATPSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 seq_shard: bool = False,
                 module_mesh: DeviceMesh = None,
                 dtype: torch.dtype = None) -> None:
        super().__init__()

        self.module_mesh = module_mesh
        self.seq_shard = seq_shard

        if self.module_mesh is None:
            self.module_mesh = get_default_mesh()

        shard_strategy_1 = [Shard(1), Shard(0)]
        shard_strategy_2 = [Shard(0), Shard(1)]

        self.attention_head_size = dim // num_heads
        self.query_key_value = ATPLinear(dim,
                                        3 * dim,
                                        self.module_mesh,
                                        shard_strategy=shard_strategy_1,
                                        input_is_shard=True,
                                        input_seq_shard=seq_shard,
                                        output_seq_shard=False,
                                        output_hidden_shard=True,
                                        params_dtype=dtype)

        self.dense = ATPLinear(dim,
                              dim,
                              self.module_mesh,
                              shard_strategy=shard_strategy_2,
                              input_is_shard=True,
                              input_hidden_shard=True,
                              input_seq_shard=False,
                              output_seq_shard=seq_shard,
                              params_dtype=dtype)

        self.dropout = nn.Dropout(dropout)

        self.core_attention = AttentionCore(self.attention_head_size, attention_dropout)

    def forward(self, x, attention_mask=None):

        qkv = self.query_key_value(x)

        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size

        local_attention_heads = num_attention_heads
        new_qkv_shape = qkv.shape[:-1] + \
            (local_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = self.core_attention(q, k, v, attention_mask)

        x = self.dense(x)
        x = self.dropout(x)

        return x