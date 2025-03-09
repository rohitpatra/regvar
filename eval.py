import json
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM
from data import prepare_datasets
from metrics import compute_metrics
from models import setup_model_and_trainer, clear_cuda, mc_dropout_predict, apply_dropout_rate, GPTNeoWithEnsembleHeads, GPTNeoWithLaplaceLastLayer
import os
from transformers import GPTNeoForCausalLM, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
import numpy as np
from torch.utils.data import DataLoader
import gc
import hydra
from omegaconf import DictConfig
import time
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)



class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Step {state.global_step}: Cleared CUDA cache and performed garbage collection.")


def combined_model_predictions(model_non_reg, model_reg, dataloader, alpha):
    model_non_reg.eval()
    model_reg.eval()
    combined_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
            outputs_non_reg = model_non_reg(inputs)
            outputs_reg = model_reg(inputs)

            # Combined prediction formula
            f_vr = alpha * (outputs_non_reg.logits - outputs_reg.logits).abs()
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * f_vr)
            combined_prob = kappa * outputs_non_reg.logits

            combined_predictions.append(combined_prob.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    # Flatten the lists
    combined_predictions = np.concatenate(combined_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return combined_predictions, true_labels


def dropout_predictions(model_non_reg, dataloader, cfg):
    model_non_reg.eval()
    combined_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batch predictions"):
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
            outputs_mean, outputs_var = mc_dropout_predict(model_non_reg, inputs, n_forward_passes=cfg.evaluate.dropout_n_samples)
            
            # Combined prediction formula
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * outputs_var)
            combined_prob = kappa * outputs_mean

            combined_predictions.append(combined_prob.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            #log.info(f'outputs_var = {outputs_var}')

    # Flatten the lists
    combined_predictions = np.concatenate(combined_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return combined_predictions, true_labels



def ensemble_predictions(model, dataloader, alpha):
    model.eval()
    combined_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batch predictions"):
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
            
            # Forward pass to get mean and variance of logits
            outputs = model(input_ids=inputs)
            logits_mean = outputs['logits_mean']
            logits_var = alpha * outputs['logits_var']

            # Probit approximation using variance (similar to dropout method)
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * logits_var)
            combined_prob = kappa * logits_mean

            combined_predictions.append(combined_prob.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    # Flatten the predictions and labels
    combined_predictions = np.concatenate(combined_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return combined_predictions, true_labels


def laplace_predictions(model, dataloader, alpha=1):
    model.eval()
    combined_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batch predictions"):
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()

            # Forward pass to get logits and hidden states
            outputs = model(input_ids=inputs, output_hidden_states=True)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            hidden_states = outputs.hidden_states[-1]  # Get the last hidden layer (Shape: [batch_size, seq_len, hidden_size])

            # Log hidden states and covariance shapes for debugging
            #print(f"hidden_states shape: {hidden_states.shape}")  # E.g., [batch_size, seq_len, hidden_size]
            #print(f"model.laplace_covariance shape: {model.laplace_covariance.shape}")  # Should be [hidden_size] for diagonal case

            # Compute the mean logits (standard forward pass logits)
            logits_mean = logits  # Shape: [batch_size, seq_len, vocab_size]

            # Compute variance from Laplace approximation
            if model.laplace_covariance.dim() == 1:  # Diagonal case
                # Element-wise multiplication of hidden states with diagonal elements of covariance
                logits_var = torch.sum((hidden_states ** 2) * model.laplace_covariance.unsqueeze(0).unsqueeze(0), dim=-1, keepdim=True) # Shape: [batch_size, seq_len, 1]
            else:
                # If covariance is full
                # einsum to efficiently compute v_{b,s} = hidden_states_{b,s,:}^T @ laplace_covariance @ hidden_states_{b,s,:}
                logits_var = torch.einsum('bsh,hk,bsk->bs', hidden_states, model.laplace_covariance, hidden_states).unsqueeze(-1)
            
            # Now compute kappa using the hidden state variance, keeping the hidden size dimension
            # kappa should have shape [batch_size, seq_len, hidden_size]
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * alpha * logits_var)  # Shape: [batch_size, seq_len, hidden_size]

            # Apply kappa to logits_mean
            combined_prob = kappa * logits_mean  # Shape: [batch_size, seq_len, vocab_size]

            combined_predictions.append(combined_prob.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    # Flatten the predictions and labels
    combined_predictions = np.concatenate(combined_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return combined_predictions, true_labels


def model_predictions(model, dataloader):
    model.eval()
    combined_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batch predictions"):
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
            outputs = model(inputs)
            combined_predictions.append(outputs.logits.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    # Flatten the lists
    combined_predictions = np.concatenate(combined_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return combined_predictions, true_labels



@hydra.main(config_path="./cfg", config_name="config")
def evaluate_model(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    gc.collect()

    datasets, tokenizer = prepare_datasets(cfg)
    log.info(f'loaded dataset/tokenizer, has keys {datasets.keys()}')

    model_reg_path = cfg.evaluate.hfl_model_path
    model_non_reg_path = cfg.evaluate.f_model_path

    # check to see if there is a lambda setting required:
    if cfg.evaluate.hfl_lambda_path is not None:
        hfl_lambda_path = os.path.join(model_reg_path, cfg.evaluate.hfl_lambda_path)
        hfl_lambda = np.load(hfl_lambda_path)
        log.info(f'using loaded hfl_lambda = {hfl_lambda} from {hfl_lambda_path}')
        cfg.model.hfl_lambda = float(hfl_lambda)

    # Load models
    if cfg.evaluate.method == 'ensemble':
        model_non_reg = GPTNeoWithEnsembleHeads.from_finetuned(
            model_non_reg_path, num_ensemble_heads=cfg.model.n_ensemble_heads
        ).to("cuda")
    elif cfg.evaluate.method == 'laplace':
        model_non_reg = GPTNeoWithLaplaceLastLayer.from_pretrained(model_non_reg_path).to("cuda")
    else:
        model_non_reg = GPTNeoForCausalLM.from_pretrained(model_non_reg_path).to("cuda")

    if cfg.evaluate.method == 'hfl':
        model_reg = GPTNeoForCausalLM.from_pretrained(model_reg_path).to("cuda")

    # Create dataloader
    test_dat = datasets[cfg.evaluate.split]
    if cfg.evaluate.subsample_test_frac < 1:
        np.random.seed(cfg.evaluate.subsample_test_seed)
        test_dat = test_dat.filter(lambda _: np.random.rand() < cfg.evaluate.subsample_test_frac)    
    
    dataloader = DataLoader(test_dat, batch_size=8) #4

    for i,var_scaling in enumerate(cfg.evaluate.var_scalings):
        start_time = time.time()  # Start the timer

        # Compute combined predictions
        if cfg.evaluate.method == 'hfl':
            alpha = var_scaling / cfg.model.hfl_lambda
            combined_predictions, true_labels = combined_model_predictions(model_non_reg, model_reg, dataloader, alpha)
            
        elif cfg.evaluate.method == 'map':
            combined_predictions, true_labels = model_predictions(model_non_reg, dataloader)
    
        elif cfg.evaluate.method == 'dropout':
            log.info(f'model config = {model_non_reg.config}') 
            model_non_reg.config.dropout_rate = cfg.evaluate.dropout_rate
            apply_dropout_rate(model_non_reg, cfg.evaluate.dropout_rate)
            model_non_reg.train()
            for name, module in model_non_reg.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    log.info(f"Dropout layer {name} with p={module.p} is active.")
            combined_predictions, true_labels = dropout_predictions(model_non_reg, dataloader, cfg)
    
        elif cfg.evaluate.method == 'ensemble':
            combined_predictions, true_labels = ensemble_predictions(model_non_reg, dataloader, var_scaling)
    
        elif cfg.evaluate.method == 'laplace':
            laplace_method = cfg.evaluate.laplace_method  # 'diag' or 'full'
            laplace_weight_prior = cfg.evaluate.laplace_weight_prior  # L2 regularization strength
            # Fit the Laplace approximation on the test data
            if cfg.evaluate.load_laplace_cov_path is not None:
                laplace_cov_load_path = os.path.join(cfg.evaluate.load_laplace_cov_path, cfg.evaluate.laplace_cov_template.format(laplace_method=cfg.evaluate.laplace_method, dataset_name=cfg.stg.target_dataset_name, dataset_version=cfg.stg.dat_version, model_name=cfg.model.name.replace('/', '-'), split=cfg.evaluate.split))
                log.info(f'loading Laplace covariance from {laplace_cov_load_path}')
                model_non_reg.load_laplace_covariance(laplace_cov_load_path)
            else:
                if i==0: 
                    log.info(f'fitting Laplace covariance to {cfg.evaluate.split}')
                    # only need to fit once if adjusting just var_scaling:
                    model_non_reg.fit_laplace_approximation(dataloader, laplace_method=laplace_method, laplace_weight_prior=laplace_weight_prior)
                    if cfg.evaluate.save_laplace_cov_path is not None:
                        model_non_reg.save_laplace_covariance(os.path.join(cfg.evaluate.save_laplace_cov_path, cfg.evaluate.laplace_cov_template.format(laplace_method=cfg.evaluate.laplace_method, dataset_name=cfg.stg.target_dataset_name, dataset_version=cfg.stg.dat_version, model_name=cfg.model.name.replace('/', '-'), split=cfg.evaluate.split)))
            combined_predictions, true_labels = laplace_predictions(model_non_reg, dataloader, alpha=var_scaling)
            
        else:
            raise NotImplementedError()

        end_time = time.time()  # End the timer
        total_time = end_time - start_time  # Calculate total wall clock time
        log.info(f"Total wall clock time for evaluate_model: {total_time:.4f} seconds")
        # Evaluate combined predictions
        eval_results = compute_metrics((combined_predictions, true_labels))        
        log.info(f"Combined Model Evaluation: {eval_results}")
        with open("evaluation_results.txt", "a") as f:
            f.write(f"Combined Model Evaluation: {eval_results}\n")
        


if __name__ == "__main__":
    clear_cuda()
    
    evaluate_model()
