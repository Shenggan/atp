import os

import torch
import torch.distributed as dist
from spmd import DeviceMesh, Shard

from tltp.layer import TLLinear


def main():

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank, world_size = dist.get_rank(), dist.get_world_size()
    print(f"DIST INFO: {rank} {world_size}")

    # can group into small mesh (e.g. we can construct mesh for each node)
    # if rank in [0,1]:
    #     mesh = DeviceMesh("cuda", [[0, 1]])
    # else:
    #     mesh = DeviceMesh("cuda", [[2, 3]])

    mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])

    shard_strategy = [Shard(1), Shard(0)]

    shard_strategy = [Shard(0), Shard(1)]

    tl_linear_1 = TLLinear(64, 64 * 4, mesh, shard_strategy, input_is_shard=False).cuda()
    tl_linear_2 = TLLinear(64 * 4, 64, mesh, shard_strategy, input_is_shard=True).cuda()

    input_ = torch.randn(2, 1024, 64).cuda().requires_grad_(True)
    output_ = tl_linear_2(tl_linear_1(input_))

    grad_output = torch.randn_like(output_).cuda()

    print(output_.shape)

    output_.backward(grad_output)

    print(input_.grad.shape)


if __name__ == '__main__':
    main()
