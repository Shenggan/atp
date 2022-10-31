import os

import torch
import torch.distributed as dist
from spmd import DeviceMesh

from tltp.layer import ColumnLinear, RowLinear


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank, world_size = dist.get_rank(), dist.get_world_size()
    print(f"DIST INFO: {rank} {world_size}")

    # NOTE: still use 2d device mesh for megatron-style TP
    mesh = DeviceMesh("cuda", [[0, 1, 2, 3]])

    tl_linear_1 = ColumnLinear(64, 64 * 4, mesh).cuda()
    tl_linear_2 = RowLinear(64 * 4, 64, mesh, input_is_shard=True).cuda()
    input_ = torch.randn(2, 1024, 64).cuda().requires_grad_(True)
    output_ = tl_linear_2(tl_linear_1(input_))

    grad_output = torch.randn_like(output_).cuda()

    print(output_.shape)

    output_.backward(grad_output)

    print(input_.grad.shape)


if __name__ == '__main__':
    main()
