import torch
import torch.distributed as dist
from spmd import Shard

from tltp.layer import TLLinear
import tltp.distributed as tltp_dist


def main():
    dist.init_process_group("nccl")
    tltp_dist.init_mesh((2, 2))
    mesh = tltp_dist.get_default_mesh()

    shard_strategy_1 = [Shard(1), Shard(0)]

    shard_strategy_2 = [Shard(0), Shard(1)]

    tl_linear_1 = TLLinear(64, 64 * 4, mesh, shard_strategy_1, input_is_shard=False).cuda()
    tl_linear_2 = TLLinear(64 * 4, 64, mesh, shard_strategy_2, input_is_shard=True).cuda()

    input_ = torch.randn(2, 1024, 64).cuda().requires_grad_(True)
    output_ = tl_linear_2(tl_linear_1(input_))

    grad_output = torch.randn_like(output_).cuda()

    print(f"output_.shape: {output_.shape}")

    output_.backward(grad_output)

    print(f"input_.grad.shape: {input_.grad.shape}")


if __name__ == '__main__':
    main()
