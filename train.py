import yaml
from models import setup_model_and_trainer, clear_cuda
from data import prepare_datasets
import metrics
import hydra
from omegaconf import DictConfig
import torch
import gc
import os
import shutil
import time
import logging

log = logging.getLogger(__name__)


def rename_checkpoint_folder(output_dir, new_path):
    for folder in os.listdir(output_dir):
        if folder.startswith('checkpoint-'):
            shutil.copytree(os.path.join(output_dir, folder), new_path)
            break


@hydra.main(config_path="./cfg", config_name="config")
def run(cfg: DictConfig) -> None:
    datasets, tokenizer = prepare_datasets(cfg)
    model, trainer = setup_model_and_trainer(cfg, datasets, tokenizer)

    start_time = time.time()  # Start the timer
    trainer.train()
    end_time = time.time()  # End the timer
    total_time = end_time - start_time  # Calculate total wall clock time
    log.info(f"Total wall clock time for training: {total_time:.4f} seconds")

    # After training, rename and move the folder
    #method_name = cfg.model.var_method
    #model_name = cfg.model.name
    #new_path = os.path.join(hydra.utils.get_original_cwd(), 'models',  f'{cfg.stg.target_dataset_name}_{cfg.stg.dat_version}_{model_name}_{method_name}')
    #rename_checkpoint_folder(cfg.model.output_dir, new_path)


if __name__ == "__main__":
    clear_cuda()
    run()
