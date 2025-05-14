#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=clara
#SBATCH --mem-per-cpu=2000

# this script was used to download the audiofiles of the podcasts

export PATH=$PATH:/path/to/ffmpeg/bin
module load Python/3.10.4-GCCcore-11.3.0

#pip install my-voice-analysis
#pip install audiofile
#pip install opensmile
#pip install pydub
pip install python-podcastindex

srun python -u download_audio.py