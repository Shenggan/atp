import torch
import torch.nn.functional as F

from tltp.layer import ColumnLinear, RowLinear
from tltp.distributed import get_default_tp_mesh


class FeedForward(torch.nn.Module):

    def __init__(self, hidden_size, ratio=4, module_mesh=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = hidden_size * ratio
        self.module_mesh = module_mesh

        if self.module_mesh is None:
            self.module_mesh = get_default_tp_mesh()

        self.dense_h_to_4h = ColumnLinear(self.hidden_size, self.ffn_hidden_size, self.module_mesh)
        self.dense_4h_to_4 = RowLinear(self.ffn_hidden_size, self.hidden_size, self.module_mesh)

        self.activation_func = F.gelu

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output
