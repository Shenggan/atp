import torch.nn as nn
import torch.nn.functional as F
from spmd import Shard

from atp.layer import ColumnLinear, RowLinear, ATPLinear
from atp.distributed import get_default_mesh


class FeedForward(nn.Module):

    def __init__(self, hidden_size, ratio=4, seq_shard=False, module_mesh=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = hidden_size * ratio
        self.module_mesh = module_mesh
        self.seq_shard = seq_shard

        if self.module_mesh is None:
            self.module_mesh = get_default_mesh()

        self.dense_h_to_4h = ColumnLinear(self.hidden_size,
                                          self.ffn_hidden_size,
                                          self.module_mesh,
                                          seq_shard=seq_shard)
        self.dense_4h_to_h = RowLinear(self.ffn_hidden_size,
                                       self.hidden_size,
                                       self.module_mesh,
                                       seq_shard=seq_shard,
                                       input_is_shard=True)

        self.activation_func = F.gelu

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class ATPFeedForward(nn.Module):

    def __init__(self, hidden_size, ratio=4, seq_shard=False, module_mesh=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = hidden_size * ratio
        self.module_mesh = module_mesh
        self.seq_shard = seq_shard

        if self.module_mesh is None:
            self.module_mesh = get_default_mesh()

        shard_strategy_1 = [Shard(1), Shard(0)]
        shard_strategy_2 = [Shard(0), Shard(1)]

        self.dense_h_to_4h = ATPLinear(self.hidden_size,
                                      self.ffn_hidden_size,
                                      self.module_mesh,
                                      shard_strategy=shard_strategy_1,
                                      input_is_shard=True,
                                      input_seq_shard=seq_shard,
                                      output_seq_shard=False)
        self.dense_4h_to_h = ATPLinear(self.ffn_hidden_size,
                                      self.hidden_size,
                                      self.module_mesh,
                                      shard_strategy=shard_strategy_2,
                                      input_is_shard=True,
                                      input_seq_shard=False,
                                      output_seq_shard=seq_shard)

        self.activation_func = F.gelu

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output
