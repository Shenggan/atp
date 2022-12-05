from setuptools import setup, find_packages

setup(
    name='atp',
    version='0.0.1',
    packages=find_packages(exclude=(
        'assets',
        'example',
        '*.egg-info',
    )),
    description=
    'Adaptive Tensor Parallelism for Large Model Traning and Inference',
)