#!/bin/bash

# Initializes the repository by pulling some of the GPU profiling repositories. Opted to use this
# script instead of git submodules since those can be a bit of a headache.
git clone https://github.com/UM-ARM-Lab/arm_segmentation.git

git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git

# Currently having trouble with the MPPI autotune pip installation.
# git clone https://github.com/UM-ARM-Lab/pytorch_mppi.git

echo "Repository initialized."
echo ""
echo "Please install the cloned repositories into your Python environment."