from datasets import load_dataset, concatenate_datasets
import numpy as np
from transformers import AutoTokenizer
import os
import torch
import logging

log = logging.getLogger(__name__)

def prepare_datasets(cfg):
    dataset_configs = cfg.datasets
    model_config = cfg.model
    stg = cfg.stg

    prepared_datasets = {}
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

    def encode(examples):
        encodings = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
        encodings['labels'] = encodings['input_ids'].copy()
        return encodings

    dataset_config = next(c for c in dataset_configs if c.name == stg.target_dataset_name)

    if stg.dat_save_template is not None:
        dat_path = stg.dat_save_template.format(dataset_name=stg.target_dataset_name, dataset_version=stg.dat_version)
        train_path = dat_path.format(split='train')
        test_path = dat_path.format(split='test')
        valid_path = dat_path.format(split='valid')
    else:
        train_path, test_path, valid_path = None, None, None

    if not(stg.force_process_dat) and (train_path is not None) and (test_path is not None) and (valid_path is not None):
        if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(valid_path):
            log.info(f"Loading preprocessed splits from file\n{train_path}\n{test_path}\n{valid_path}")
            train_dataset = torch.load(train_path)
            test_dataset = torch.load(test_path)
            valid_dataset = torch.load(valid_path)
            prepared_datasets = {
                'train': train_dataset,
                'test': test_dataset,
                'validation': valid_dataset
            }
            log.info(f'train size = {len(prepared_datasets["train"])}, test size = {len(prepared_datasets["test"])}, validation size = {len(prepared_datasets["validation"])}')
            return prepared_datasets, tokenizer

    # Load and preprocess the dataset if not loading from saved files
    hf_split = dataset_config.get('split', 'train')
    if 'version' in dataset_config:
        dataset = load_dataset(dataset_config.name, dataset_config.version, ignore_verifications=True)
    else:
        dataset = load_dataset(dataset_config.name, ignore_verifications=True)

    log.info(f'loaded {dataset_config.name, dataset_config.get("version",None)}, has keys {dataset.keys()}')

    len_train = len(dataset['train'])
    #if 'sample_fraction' in dataset_config:
    if len_train > cfg.stg.target_dat_size:
        subsample_rate = cfg.stg.target_dat_size / len_train
        np.random.seed(dataset_config.get('random_seed', 0))
        dataset = dataset.filter(lambda _: np.random.rand() < subsample_rate)

    if 'preprocess' in dataset_config:
        dataset = dataset.map(dataset_config['preprocess'], batched=True)
    else:
        dataset = dataset.map(encode, batched=True)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    if dataset_config.get('pool_and_split',False) or ('validation' not in dataset.keys()): 
        log.info(f'pooling and splitting')
        dataset = concatenate_datasets([split for split in dataset.values()])
        # Randomly split the dataset into train, test, and validation sets
        train_size = int(dataset_config.get('train_split_fraction', 0.8) * len(dataset))
        test_size = int(dataset_config.get('test_split_fraction', 0.1) * len(dataset))
        valid_size = int(dataset_config.get('valid_split_fraction', 0.1) * len(dataset))
        seed = dataset_config.get('random_seed', 0)
    
        train_testvalid = dataset.train_test_split(test_size=test_size + valid_size, seed=seed)
        test_valid = train_testvalid['test'].train_test_split(test_size=valid_size, seed=seed)
    
        prepared_datasets = {
            'train': train_testvalid['train'],
            'test': test_valid['train'],
            'validation': test_valid['test']
        }
    else:
        prepared_datasets = dataset
    
    # Save the splits if a template is provided
    if train_path is not None and test_path is not None and valid_path is not None:
        log.info("Saving preprocessed splits to file...")
        torch.save(prepared_datasets['train'], train_path)
        torch.save(prepared_datasets['test'], test_path)
        torch.save(prepared_datasets['validation'], valid_path)

    log.info(f'train size = {len(prepared_datasets["train"])}, test size = {len(prepared_datasets["test"])}, validation size = {len(prepared_datasets["validation"])}')

    return prepared_datasets, tokenizer
