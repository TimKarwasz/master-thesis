#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -t 2-0
#SBATCH --partition=clara
#SBATCH --mem-per-cpu=24000

# this script was used to extract the audiofeatures

export PATH=$PATH:/path/tp/ffmpeg/bin
module load Python/3.10.4-GCCcore-11.3.0

pip install my-voice-analysis
pip install audiofile
pip install opensmile
pip install pydub

srun python -u get_audiofeatures.py