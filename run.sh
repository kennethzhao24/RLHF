export TRITON_CACHE_DIR="/u/yzhao25/triton_cache"

# python nemo_train.py

python nemo_lora.py

# apptainer run --nv --bind /work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface /u/yzhao25/nemo.sif /bin/bash --login