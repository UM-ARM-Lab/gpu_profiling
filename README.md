# ARM Lab GPU Profiling

This repo houses the code the UMich ARM Lab uses to benchmark their GPUs.

# Installation

## Hardware

These benchmarks require a CUDA-capable GPU, meaning a somewhat recent NVIDIA GPU and driver.

## Software Dependencies

The only software dependencies for running these benchmarks is Python 3 and `pipenv`. Python 3 should be on your path as `python3`. Your system installation of Python will is sufficient as the benchmarking functions handle virtual environment creation for you. Additionally, `pipenv` should be installed and on your path.

If `pipenv` is not installed, install it with `python3 -m pip install pipenv`.

## Installation And Initialization Process

Initialize the repository with:

```bash
./init_repo.sh
```

The script clones the necessary repositories and creates a `pipenv` environment for running the benchmarks. Note that this will ask you which CUDA version you would like for PyTorch to be installed with.

# Usage

## Running Benchmarks

1. Ensure that you have initialized the repository by following instructions in the [Installation](#installation) section.
2. `cd` to where this repository is located.
3. Activate the `pipenv` environment with `pipenv shell`.
4. Run the benchmarks with `python run_benchmarks.py`.

## Comparison With Other Rigs

To compare the results of your machine with others, run `python compare_results.py`.

We include the profiling results of a few of our older computers for reference.

However, the comparison script isn't all-encompassing so if you want fine-grained control of plotting, you may desire to make your own script or notebook.