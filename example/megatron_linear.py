import torch
import torch.distributed as dist

from atp.layer import ColumnLinear, RowLinear
import atp.distributed as atp_dist


def main():
    dist.init_process_group("nccl")
    # NOTE: still use 2d device mesh for megatron-style TP
    atp_dist.init_mesh([1, 2])

    rank = dist.get_rank()
    mesh = atp_dist.get_default_mesh()

    atp_linear_1 = ColumnLinear(64, 64 * 4, mesh, seq_shard=True).cuda()
    atp_linear_2 = RowLinear(64 * 4, 64, mesh, input_is_shard=True, seq_shard=True).cuda()
    input_ = torch.randn(2, 512, 64).cuda().requires_grad_(True)
    output_ = atp_linear_2(atp_linear_1(input_))

    grad_output = torch.randn_like(output_).cuda()

    print(f"[{rank}] output_.shape: {output_.shape}")

    output_.backward(grad_output)

    print(f"[{rank}] input_.grad.shape: {input_.grad.shape}")

    dist.all_reduce(input_.grad)

    print(f"[{rank}] input_.grad.shape: {input_.grad.shape}")


if __name__ == '__main__':
    main()
