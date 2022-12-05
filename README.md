![](./assert/banner.png)

# ATP: Adaptive Tensor Parallelism

Adaptive Tensor Parallelism for Large Model Traning and Inference

ATP provides a high-performance implementation of Topology-aware Tensor Parallelism with the following characteristics.

1. Two-Level Search Space for Tensor Parallelism.
2. Adaptive Tensor Parallelism with Hierarchical Communication Matrix.
3. Chunk-based Communication-Computation Overlapping.
4. An estimator that helps study the performance of ATP on networks with different topologies.

## Installation

To install ATP, you will need:

+ Python 3.8 or 3.9.
+ PyTorch 1.13
+ SPMD from [pytorch/tau](https://github.com/pytorch/tau)

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install git+https://github.com/pytorch/tau.git@89700fd
```

## Usage

