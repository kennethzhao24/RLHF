import os
import json
import random
import pandas as pd
random.seed(42)

from pathlib import Path

import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed



PREPARED_DATA_DIR = '/u/yzhao25/RLHF/data/prepared-data'


# Define directories for intermediate artifacts
NEMO_MODELS_CACHE = "/nemo-experiments/models-cache"
NEMO_DATASETS_CACHE = "/nemo-experiments/data-cache"

os.environ["NEMO_DATASETS_CACHE"] = NEMO_DATASETS_CACHE
os.environ["NEMO_MODELS_CACHE"] = NEMO_MODELS_CACHE


# Configure the number of GPUs to use
NUM_GPU_DEVICES = 1


from getpass import getpass
from huggingface_hub import login

login(token=getpass("Input your HF Access Token"))


# You can just as easily swap out the model with the 120B variant, or execute this on a remote cluster.

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=run.Config(llm.GPTOSSModel, llm.GPTOSSConfig20B),
        source="hf:///nemo-experiments/models/gpt-oss-20b",
        overwrite=False,
    )
    
# Run your experiment locally
run.run(configure_checkpoint_conversion(), executor=run.LocalExecutor())



recipe = llm.gpt_oss_20b.finetune_recipe(
    name="gpt_oss_20b_finetuning",
    dir="/nemo-experiments/",
    num_nodes=1,
    num_gpus_per_node=NUM_GPU_DEVICES,
    peft_scheme='lora',  # 'lora', 'none' (for SFT)
)


from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

dataloader = run.Config(
        FineTuningDataModule,
        dataset_root=PREPARED_DATA_DIR,
        seq_length=2048,
        micro_batch_size=4,
        global_batch_size=64
    )

# Configure the recipe
recipe.data = dataloader


LOG_DIR = "/nemo-experiments/results"
LOG_NAME = "nemo2_gpt_oss_sft_customer_ticket_routing"

def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=200,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return run.Config(
        nl.NeMoLogger,
        name=LOG_NAME,
        log_dir=LOG_DIR,
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None,
    )

recipe.log = logger()


def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(
            nl.RestoreConfig, path=f"nemo:///{NEMO_MODELS_CACHE}/gpt-oss-20b"
        ),
        resume_if_exists=True,
    )
    
recipe.resume = resume()


recipe.trainer.max_steps = 100
recipe.trainer.val_check_interval = 25
recipe.trainer.limit_val_batches = 2
recipe.optim.config.lr = 2e-4


run.run(recipe, executor=run.LocalExecutor(ntasks_per_node=NUM_GPU_DEVICES))