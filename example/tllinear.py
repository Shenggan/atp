import torch
import torch.distributed as dist
from spmd import Shard

from atp.layer import ATPLinear
import atp.distributed as atp_dist


def main():
    dist.init_process_group("nccl")
    atp_dist.init_mesh((2, 2))
    mesh = atp_dist.get_default_mesh()

    shard_strategy_1 = [Shard(1), Shard(0)]

    shard_strategy_2 = [Shard(0), Shard(1)]

    atp_linear_1 = ATPLinear(2048,
                           2048 * 4,
                           mesh,
                           shard_strategy_1,
                           input_is_shard=False,
                           input_seq_shard=False).cuda()
    atp_linear_2 = ATPLinear(2048 * 4,
                           2048,
                           mesh,
                           shard_strategy_2,
                           input_is_shard=True,
                           output_seq_shard=True).cuda()

    atp_linear_3 = ATPLinear(2048,
                           2048 * 4,
                           mesh,
                           shard_strategy_1,
                           input_is_shard=True,
                           input_seq_shard=True).cuda()
    atp_linear_4 = ATPLinear(2048 * 4,
                           2048,
                           mesh,
                           shard_strategy_2,
                           input_is_shard=True,
                           output_seq_shard=True).cuda()

    atp_linear_5 = ATPLinear(2048,
                           2048 * 4,
                           mesh,
                           shard_strategy_1,
                           input_is_shard=True,
                           input_seq_shard=True).cuda()
    atp_linear_6 = ATPLinear(2048 * 4,
                           2048,
                           mesh,
                           shard_strategy_2,
                           input_is_shard=True,
                           output_seq_shard=True).cuda()

    input_ = torch.randn(4, 1024, 2048).cuda().requires_grad_(True)

    def test_loop():
        output_ = atp_linear_2(atp_linear_1(input_))
        output_ = atp_linear_4(atp_linear_3(output_))
        output_ = atp_linear_6(atp_linear_5(output_))

        grad_output = torch.randn_like(output_).cuda()

        print(f"output_.shape: {output_.shape}")

        output_.backward(grad_output)

        print(f"input_.grad.shape: {input_.grad.shape}")

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=3, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/atplinear'),
        record_shapes=False,
        with_stack=False)

    prof.start()
    torch.cuda.synchronize()
    for _ in range(10):
        test_loop()
        torch.cuda.synchronize()
        prof.step()
    prof.stop()


if __name__ == '__main__':
    main()
