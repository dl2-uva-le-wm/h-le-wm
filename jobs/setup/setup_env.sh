#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --chdir=jobs/setup
#SBATCH --output=out/install_env_%j.out
#SBATCH --error=out/install_env_%j.err

module purge
module load 2025
module load Anaconda3/2025.06-1

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

cd $HOME/h-lewm

conda env create -f environment-gpu.yml
