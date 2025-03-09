import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import norm
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


def compute_metrics(eval_pred, batch_size=32):
    logits, labels = eval_pred

    # Flatten the labels (keeping them on CPU)
    labels = np.array(labels).flatten()

    # Initialize an empty list to store batch predictions
    all_predictions = []
    
    # Process logits in batches (staying on CPU until batch is processed)
    for i in range(0, logits.shape[0], batch_size):
        # Extract the batch from logits (keeps this batch on CPU)
        logits_batch = torch.tensor(logits[i:i + batch_size])
        
        # Move the current batch to GPU
        logits_batch = logits_batch.to('cuda')
        
        # Apply softmax to the batch
        predictions_batch = torch.nn.functional.softmax(logits_batch, dim=-1)
        
        # Reshape predictions to match the labels (within the batch)
        predictions_batch = predictions_batch.view(-1, predictions_batch.shape[-1])
        
        # Move the predictions back to CPU and convert to numpy
        predictions_batch = predictions_batch.cpu().numpy()
        
        # Append the predictions to the list
        all_predictions.append(predictions_batch)
    
    # Concatenate all predictions into a single numpy array
    all_predictions = np.vstack(all_predictions)

    # Calculate the Negative Log-Likelihood
    log.info(f'calculating negative log lik')
    nll = compute_nll_ci(all_predictions, labels)

    # Calculate ECE
    log.info(f'calculating ECE')
    ece = compute_ece_ci(all_predictions, labels)
    
    return {'nll': nll, 'ece': ece}


def compute_nll_ci(probs, labels):
    true_probs = probs[np.arange(len(labels)), labels]
    
    # Compute log probabilities
    log_probs = np.log(true_probs + 1e-10)  # small epsilon to prevent log(0)
    mean_nll = -np.mean(log_probs)
    std_error = np.std(log_probs) / np.sqrt(len(log_probs))
    
    # 95% CI assuming the central limit theorem
    ci_bounds = norm.ppf([0.025, 0.975])
    ci_size = ci_bounds[1] * std_error
    
    return mean_nll, ci_size


def compute_ece_ci(predictions, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_error = 0.0
    bin_counts = []
    bin_error = np.zeros(n_bins)
    bin_preds_mean = np.zeros(n_bins)
    bin_preds_var = np.zeros(n_bins)

    for bin_index in tqdm(range(n_bins), desc="ECE Bins"):        
        bin_lower = bin_boundaries[bin_index]
        bin_upper = bin_boundaries[bin_index + 1]
        in_bin = (predictions.max(axis=1) > bin_lower) & (predictions.max(axis=1) <= bin_upper)
        if in_bin.sum() == 0:
            continue

        # Assuming label prediction is the class with max probability
        bin_preds = predictions[in_bin]
        bin_labels = labels[in_bin]
        bin_pred_probs = bin_preds[np.arange(bin_preds.shape[0]), bin_preds.argmax(axis=1)]
        bin_true_probs = np.mean(bin_labels == bin_preds.argmax(axis=1))
        bin_preds_mean[bin_index] = bin_pred_probs.mean()
        bin_preds_var[bin_index] = bin_pred_probs.var()

        bin_error[bin_index] = np.abs(bin_true_probs - np.mean(bin_pred_probs))
        ece += bin_error[bin_index] * in_bin.sum()
        total_error += (bin_error[bin_index] ** 2) * in_bin.sum()
        bin_counts.append(in_bin.sum())

    log.info(f'ECE bin_mean = {list(bin_preds_mean)}')
    log.info(f'ECE bin_var = {list(bin_preds_var)}')
    log.info(f'ECE bin_counts = {bin_counts}')
    log.info(f'ECE bin_error = {list(bin_error)}')
    
    mean_ece = ece / sum(bin_counts)
    variance_ece = total_error / sum(bin_counts)**2
    std_error_ece = np.sqrt(variance_ece)

    # 95% CI using normal approximation
    ci_bounds = norm.ppf([0.025, 0.975])
    ci_size = ci_bounds[1] * std_error_ece

    return mean_ece, ci_size


def save_results(results, output_dir):
    with open(f"{output_dir}/evaluation_results.txt", "w") as file:
        file.write(str(results))
