import torch

from .fake_comm_kernel import FakeCommKernel

WORLD_SIZE = 1
ESTIMATOR_DEBUG = False

FAKE_COMM_KERNEL = FakeCommKernel()

if ESTIMATOR_DEBUG:
    FAKE_COMM_KERNEL.print_summary()


def init(world_size):
    global WORLD_SIZE
    WORLD_SIZE = world_size


class FakeWork(object):

    def __init__(self, event=None):
        self.event = event

    def wait(self):
        if self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)


def all_reduce(tensor, op=None, group=None, async_op=False):
    # calculate the communication time
    sleep_in_us = 10
    # calculate the output tensor_size
    output_tensor = tensor
    event = FAKE_COMM_KERNEL.sleep(sleep_in_us, async_op=async_op)
    if event is not None:
        work = FakeWork(event)
        return tensor, work
    return output_tensor
