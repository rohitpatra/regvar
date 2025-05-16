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
from typing import List, Tuple
from torch.func import make_functional_with_buffers, jvp


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


def _make_u_tuples(
    model: torch.nn.Module,
    sketch_size: int,
    seed: int,
    std: float,
    device: torch.device | str = "cpu",
) -> List[Tuple[torch.Tensor, ...]]:
    """
    Create `sketch_size` tuples of per-parameter Gaussian noise.
    Stored on `device` (default CPU) to save GPU memory.
    """
    names = [n for n, _ in model.named_parameters()]
    tuples: list[Tuple[torch.Tensor, ...]] = []

    gen = torch.Generator(device=device)
    for k in range(sketch_size):
        gen.manual_seed(seed + k)
        u_dict = {
            n: torch.randn(p.shape, dtype=p.dtype, device=device, generator=gen) * std
            for n, p in model.named_parameters()
        }
        tuples.append(tuple(u_dict[n] for n in names))

    return tuples


@torch.no_grad()
def sketching_predictions(
    model_non_reg: torch.nn.Module,
    model_reg_list: List[torch.nn.Module],
    dataloader,
    cfg: dict,
    *,
    alpha: float = 1.0,
):
    """
    Memory-light version: streams one JVP per sketch to keep GPU usage low.
    Returns (combined_predictions, true_labels) as NumPy arrays.
    """
    device = next(model_non_reg.parameters()).device

    # put all models in eval mode
    model_non_reg.eval()
    for m in model_reg_list:
        m.eval()

    # make functional copy once
    func_non_reg, params_non_reg, buffers_non_reg = make_functional_with_buffers(
        model_non_reg
    )

    # ── sketch config ────────────────────────────────────────────────────────
    m_cfg = cfg.model
    sketch_size = m_cfg.right_sketch_size
    total_params = sum(p.numel() for p in model_non_reg.parameters() if p.requires_grad)
    target_std = (1.0 / total_params) ** 0.5 if total_params else 1.0
    sketching_multiplier = m_cfg.sketching_multiplier
    target_std *= sketching_multiplier
    

    # keep noise on CPU to avoid GPU OOM
    u_tuples_cpu = _make_u_tuples(
        model_non_reg, sketch_size, m_cfg.u_seed_init, target_std, "cpu"
    )

    # helper: functional forward that only depends on params
    def f_params(p, inp):
        return func_non_reg(p, buffers_non_reg, input_ids=inp).logits  # (B,C,V)

    preds, lbls = [], []
    
    for batch in tqdm(dataloader, desc="Sketching"):
        # dataloader gives us a dict
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits_nr = model_non_reg(inputs).logits  # (B,C,V)

        # stack diff over regularized models → shape (S,B,C,V)
        logits_diff = torch.stack(
            [logits_nr - m(inputs).logits for m in model_reg_list], dim=0
        )

        # incremental variance: start at 0
        var_estimate = torch.zeros_like(logits_nr)  # (B,C,V)

        for k, u_cpu in enumerate(u_tuples_cpu):
            # move one noise tuple to GPU
            u = tuple(t.to(device, non_blocking=True) for t in u_cpu)

            # JVP -> (outputs, jvp_out); we keep only jvp_out
            _, jvp_out = jvp(
                lambda p: f_params(p, inputs),
                (params_non_reg,),
                (u,),
            )

            var_estimate += logits_diff[k] * jvp_out  # (B,C,V)

            del jvp_out, u
            torch.cuda.empty_cache()  # helps fragmentation on long runs
        var_estimate_adjusted_for_sketching_multiplier = var_estimate / (sketching_multiplier**2)

        f_vr = alpha * var_estimate_adjusted_for_sketching_multiplier.abs()
        kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8.0 * f_vr)
        combined_prob = kappa * logits_nr

        preds.append(combined_prob.detach().cpu().numpy())
        lbls.append(labels.detach().cpu().numpy())

        # cleanup to keep peak GPU memory usage low
        del logits_nr, logits_diff, var_estimate, combined_prob, labels, inputs
        torch.cuda.empty_cache()
        gc.collect()

    return np.concatenate(preds), np.concatenate(lbls)


def sketching_predictions_alternate(model_non_reg, model_reg_list, dataloader, cfg, alpha=1):
    model_config = cfg.model
    log.info(f"Starting sketching predictions with alpha={alpha}")
    
    # Set device
    device = next(model_non_reg.parameters()).device
    
    # Ensure all models are in eval mode
    model_non_reg.eval()
    func_model_non_reg, params_non_reg, buffers_non_reg = make_functional_with_buffers(model_non_reg.eval())

    for model_reg in model_reg_list:
        model_reg.eval()
    
    combined_predictions = []
    true_labels = []
    
    # Generate sketch vectors based on seed
    u_seed_starter = model_config.u_seed_init
    sketch_size = model_config.right_sketch_size
    
    # Calculate total number of trainable parameters
    total_params = sum(p.numel() for p in model_non_reg.parameters() if p.requires_grad)
    log.info(f"Total trainable parameters: {total_params}")

    # Calculate desired standard deviation for variance = 1 / total_params
    if total_params > 0:
        target_std_dev = (1.0 / total_params)**0.5
        log.info(f"Initial target standard deviation for u_eval generation: {target_std_dev}")
    else:
        target_std_dev = 1.0 # Avoid division by zero, fallback to std dev 1
        log.warning("Model has no trainable parameters. Using std dev 1 for u_eval.")

    # Apply sketching_multiplier if provided
    sketching_multiplier = model_config.sketching_multiplier
    if sketching_multiplier is not None:
        target_std_dev *= sketching_multiplier
        log.info(f"Applied sketching_multiplier {sketching_multiplier}, adjusted target_std_dev: {target_std_dev}")
    
    # Process batches one at a time
    for i, batch in enumerate(tqdm(dataloader, desc="Sketching batch predictions")):
        # Clear cache at strategic points only
        if i % 5 == 0:  # Less frequent clearing (was 3)
            torch.cuda.empty_cache()
            gc.collect()
            log.info(f"Processing batch {i}/{len(dataloader)} - Cleared CUDA cache")
        
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        
        # Compute logits differences between non-regularized and regularized models one by one
        logits_diff_list = []
        output_non_reg = model_non_reg(inputs)
        logits_non_reg = output_non_reg.logits
        
        for model_reg in model_reg_list:
            outputs_reg = model_reg(inputs)
            logits_reg = outputs_reg.logits
            logits_diff = logits_non_reg - logits_reg
            logits_diff_list.append(logits_diff)
            # Free memory of individual model outputs
            del logits_reg, outputs_reg
        
        # Stack the differences into a single tensor [sketch_size, batch_size, seq_len, vocab_size]
        logits_diff_tensor = torch.stack(logits_diff_list, dim=0)
        # Free memory of the list but not the tensor we just created
        del logits_diff_list
        
        jvp_out_list = []
        # Compute the JVPs for each sketch
        # helper that only depends on params
        def f_params(p):
            return func_model_non_reg(p, buffers_non_reg, input_ids=inputs).logits     # (B, Context, V)
        
        # Only clear cache once before the memory-intensive JVP calculations
        torch.cuda.empty_cache()
        
        for sketch_number in range(sketch_size):
            u_local_seed = u_seed_starter + sketch_number
            log.info(f"Setting torch seed to {u_local_seed} for reproducible u_eval generation for sketch number {sketch_number}")
            torch.manual_seed(u_local_seed)
            
            # Generate random noise vector u_eval for each parameter with same shape
            u_eval = {}
            for name, param in model_non_reg.named_parameters():
                # Generate standard normal noise and scale it by the target standard deviation
                u_eval[name] = (torch.randn_like(param) * target_std_dev).to(device)
            u_tuple = tuple(u_eval[n] for n, _ in model_non_reg.named_parameters())
            # compute corresponding jvp
            _, jvp_out = jvp(f_params, (params_non_reg,), (u_tuple,))
            # jvp_out is of shape [batch_size, seq_len, vocab_size]
            jvp_out_list.append(jvp_out)
            # Clean up u_eval to save memory
            del u_eval, u_tuple
            
            # Only clear cache periodically during JVP calculations if needed
            if sketch_number > 0 and sketch_number % (sketch_size // 2) == 0:
                torch.cuda.empty_cache()
        
        # Stack all JVP outputs
        jvp_out_tensor = torch.stack(jvp_out_list, dim=0)
        # Free memory
        del jvp_out_list
        
        # Log memory usage for debugging
        log.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        log.info(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        
        # Using einsum for more explicit dimension handling
        var_estimate = torch.einsum('sbcv,sbcv->bcv', logits_diff_tensor, jvp_out_tensor)
        var_estimate_adjusted_for_sketching_multiplier = var_estimate / (sketching_multiplier**2)

        
        # Free memory of tensors no longer needed
        del logits_diff_tensor, jvp_out_tensor
        
        f_vr = alpha * torch.abs(var_estimate_adjusted_for_sketching_multiplier)
        kappa = 1 / torch.sqrt(1. + np.pi / 8 * f_vr)
        combined_prob = kappa * logits_non_reg
        
        # Convert to numpy immediately to free GPU memory
        combined_prob_np = combined_prob.cpu().detach().numpy()
        true_labels_np = labels.cpu().numpy()
        
        # Store the predictions and labels
        combined_predictions.append(combined_prob_np)
        true_labels.append(true_labels_np)
        
        # Free remaining tensors - do a single cleanup at the end of batch processing
        del var_estimate, f_vr, kappa, combined_prob, inputs, labels, logits_non_reg, output_non_reg
        
        # Only clear cache at the end of batch processing
        torch.cuda.empty_cache()
        gc.collect()
    
    # Flatten the lists at the end of processing all batches
    combined_predictions = np.concatenate(combined_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    log.info(f"Completed sketching predictions. Shape: {combined_predictions.shape}")
    
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
