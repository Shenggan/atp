import torch

import triton
import triton.language as tl
import triton._C.libtriton.triton as _triton


@triton.jit
def _sleep_triton_kernel(
    delay_clocks,
    start_clock_pointer,
    BLOCK_SIZE: tl.constexpr,
):
    tl.atomic_min(start_clock_pointer, tl.clock())
    start_clock = tl.load(start_clock_pointer)
    while tl.clock() - start_clock < delay_clocks:
        pass


class FakeCommKernel(object):

    def __init__(self):

        self.start_clock_pointer = torch.zeros(1, dtype=torch.int64, device='cuda')
        self.int64_max = torch.iinfo(torch.int64).max

        # the setting measured in NVIDIA A100
        # need `nvidia-smi -lgc 1410` to fix the clock ratio to control sleep time accurately
        self.overhead_const_ = 3
        self.overhead_ratio_ = 0.99

        self.device = torch.cuda.current_device()
        self.triton_rt_backend = _triton.runtime.backend.CUDA
        # _triton.runtime.clock_rate return in ms, self.clock_rate in us
        self.clock_rate = _triton.runtime.clock_rate(self.triton_rt_backend, self.device) // 1000

        # triton kernel setting
        self.kernel = _sleep_triton_kernel
        self.kernel_grid = (16,)
        self.kernel_block_size = 256

        # warm up triton kernel
        self.__warm_up()

    def print_summary(self):
        print(f"[FakeCommKernel]: ")
        print(f"  self.clock_rate = {self.clock_rate} ( in us )")
        print(f"  self.overhead_const_ = {self.overhead_const_}")
        print(f"  self.overhead_ratio_ = {self.overhead_ratio_}")

    def __warm_up(self, sleep_in_us=100, times=2):
        for _ in range(times):
            self.sleep(sleep_in_us)

    def __prep_sleep(self):
        self.start_clock_pointer[0] = self.int64_max

    def __calculate_clock(self, time_in_us):
        time_in_us = time_in_us * self.overhead_ratio_ - self.overhead_const_
        return time_in_us * self.clock_rate

    def sleep(self, sleep_in_us):
        self.__prep_sleep()
        self.kernel[self.kernel_grid](self.__calculate_clock(sleep_in_us),
                                      self.start_clock_pointer,
                                      BLOCK_SIZE=self.kernel_block_size)
