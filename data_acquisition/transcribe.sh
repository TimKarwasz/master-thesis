#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -t 2-0
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem-per-gpu=11000

# this script was used to transcribe every podcast on the scientific computing cluster from the university of leipzig

day_number="14"

# setup needed variables
BASE_DIR="/work/tk81fevo-whisper2/tk81fevo-whisper-transcription-1739240423/"
DATA_DIR="data/podcasts/"
OUTPUT_DIR="data/archive/2024-12-"$day_number"_00-00_de/transcriptions/"

# archive dir where files should get moved too so they do not get processed twice / needs to be changed for every run
ARCHIVE_DIR="data/archive/2024-12-"$day_number"_00-00_de/podcasts/"

export PATH=$PATH:/work/tk81fevo-whisper2/tk81fevo-whisper-transcription-1739240423/misc/ffmpeg-master-latest-linux64-lgpl/bin
module load Python/3.10.4-GCCcore-11.3.0

pip install -U openai-whisper

cd $BASE_DIR

python --version
# source whisper/bin/activate mnight not be needed

array=(${BASE_DIR}${DATA_DIR}*)
start_whisper_time=$(date +%s)

GPU_ID=0 # either via gpu id or slurm will do this for us i guess

# [tiny, base, small, medium, large-v2, large-v3, turbo]
MODEL_SIZE=turbo

# [transcribe, translate]
TASK=transcribe

# Log file
LOG_FILE=${BASE_DIR}log.log

for i in "${array[@]}"
do
    DURATION_IN_SECOND=`ffprobe -i $i -show_entries format=duration -v quiet -of csv="p=0"`
    START_TIME=$(date '+%m/%d/%Y %H:%M:%S')
    START_IN_SECONDS=$(date --date "$START_TIME" +%s)
    SIZE_IN_BYTES=$(wc --bytes < $i)

    whisper --device cuda $i --model ${MODEL_SIZE} --language de \
            --output_format json --task $TASK \
            --output_dir ${BASE_DIR}${OUTPUT_DIR} \
            --condition_on_previous_text False

    
    FINISHED_TIME=$(date '+%m/%d/%Y %H:%M:%S')
    FINISHED_IN_SECONDS=$(date --date "$FINISHED_TIME" +%s)
    TIME_DIFF=$((FINISHED_IN_SECONDS - START_IN_SECONDS))
    TIME_RATIO=$(perl -e "print $DURATION_IN_SECOND/$TIME_DIFF") # we have to see if this works on the cluster
    echo $TASK $MODEL_SIZE $i $DURATION_IN_SECOND $TIME_DIFF $TIME_RATIO $SIZE_IN_BYTES >> "$LOG_FILE"

    # delete file i here ? would save a lot of space ; cant, we need it for other metadata extraction
    mv $i ${BASE_DIR}${ARCHIVE_DIR}

done
end_whisper_time=$(date +%s)
echo $(( end_whisper_time - start_whisper_time )) >> "$LOG_FILE"

# deactivate