import torch
from atp.utils import FakeCommKernel

PROFILE = False
PROFILE_WAIR = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 3


def test_loop(x, y, fake_comm_kernel, sleep_in_us=200):
    z1 = x + y
    fake_comm_kernel.sleep(sleep_in_us)  # sleep 200 us
    print(f"sleep {sleep_in_us} us!")
    z2 = x * y
    return z1, z2


def main():
    x = torch.rand(10240, device='cuda')
    y = torch.rand(10240, device='cuda')
    fake_comm_kernel = FakeCommKernel()

    fake_comm_kernel.print_summary()

    if PROFILE:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
            record_shapes=False,
            with_stack=False)

        prof.start()

    torch.cuda.synchronize()
    for _ in range(PROFILE_WAIR + PROFILE_WARMUP + PROFILE_ACTIVE):
        _, _ = test_loop(x, y, fake_comm_kernel)
        torch.cuda.synchronize()
        if PROFILE:
            prof.step()
    if PROFILE:
        prof.stop()

    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
