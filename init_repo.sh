#!/bin/bash

echo "Cloning benchmark repositories"
echo "------------------------------"
# Initializes the repository by pulling some of the GPU profiling repositories. Opted to use this
# script instead of git submodules since those can be a bit of a headache.
if [ ! -d "arm_segmentation" ]; then
  git clone https://github.com/UM-ARM-Lab/arm_segmentation.git
else
    echo "Detected arm_segmentation already cloned"
fi

if [ ! -d "pytorch_kinematics" ]; then
    git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git
else
    echo "Detected pytorch_kinematics already installed"
fi

# Currently having trouble with the MPPI autotune pip installation.
# git clone https://github.com/UM-ARM-Lab/pytorch_mppi.git
echo ""

echo "Creating Python virtual environment (pipenv) file"
echo "-------------------------------------------------"
# This Python file only uses standard libraries so it doesn't need to be ran in any sort of
# environment. As long as the user has a Python 3 interpreter installed and on their path, they'll
# be fine.
python3 populate_pipenv_template.py
echo ""

# Check if pipenv is installed.
if ! command -v pipenv &> /dev/null
then
    echo "pipenv could not be found. Please install with:"
    echo '    `python3 -m pip install pipenv`'
    echo "and run:"
    echo '    `pipenv install`'
    echo "in this repositories directory."
    exit 1
fi

# TODO: If not installed, error out and ask user to install pipenv themself.

echo "Creating pipenv environment if necessary"
echo "----------------------------------------"
pipenv install
echo ""


echo "Repository initialized!"
echo "You can now run the benchmarks. Simply activate the pipenv environment with:"
echo '    `pipenv shell`'
echo "and run:"
echo '    `python run_benchmarks.py`'