import numpy as np
from transformers import AutoTokenizer, GPTNeoForCausalLM, Trainer, TrainingArguments, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian
from torch.autograd import grad
from metrics import compute_metrics
import gc
import logging
from box import Box
import os
import random


log = logging.getLogger(__name__)


def clear_cuda():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
class HFLTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, lam=0.1, **kwargs):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.tokenizer = tokenizer
        self.lam = lam

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Compute additional loss term as the mean of the logits scaled by `lam`
        additional_loss = self.lam * logits.abs().mean()

        # Total loss is the sum of the standard loss and the additional term
        total_loss = loss + additional_loss
        return (total_loss, outputs) if return_outputs else total_loss

class HFLOracleTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, lam=0.1, u_eval=None, **kwargs):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.tokenizer = tokenizer
        self.lam = lam
        self.u_eval = u_eval
        # assert u_eval is a dictionary with the same keys as the model parameters and same shape
        u_param_names = set(name for name, _ in model.named_parameters())
        assert set(u_eval.keys()) == u_param_names
        for name, param in model.named_parameters():
            assert u_eval[name].shape == param.shape
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))        

        # Compute dot product with model parameters (theta' u)
        dot_product = 0 
        for name, param in model.named_parameters():
            dot_product += (param * self.u_eval[name]).sum()
        additional_loss = self.lam * dot_product

        # Total loss is the sum of the standard loss and the additional term
        total_loss = loss + additional_loss
        return (total_loss, outputs) if return_outputs else total_loss


class EnsembleTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """
        Overriding the default save_model method to ensure all model components (including ensemble heads)
        are saved properly.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self._save(output_dir)

    def _save(self, output_dir):
        """
        Ensure that both the base model and the custom ensemble heads are saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the model (base + ensemble heads)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save trainer state
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


class GPTNeoForCausalLMWithDropout(GPTNeoForCausalLM):
    def apply_dropout_rate(self, dropout_rate):
        """
        Recursively apply a new dropout rate to all layers in the transformer block that use dropout.
        """
        for name, module in self.named_modules():
            # Identify and modify dropout layers within the transformer blocks
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate

    def forward(self, *args, **kwargs):
        mc_dropout = kwargs.pop('mc_dropout', False)
        # Set model to train mode (activates dropout) if mc_dropout is True during inference
        if mc_dropout:
            self.train()
        else:
            self.eval()

        return super().forward(*args, **kwargs)



class GPTNeoWithEnsembleHeads(GPTNeoForCausalLM):
    def __init__(self, config, num_ensemble_heads=1, ensemble_start_layer=None):
        super().__init__(config)
        self.num_ensemble_heads = num_ensemble_heads
        
        # ensemble_start_layer defines where the ensemble heads start to diverge
        self.ensemble_start_layer = ensemble_start_layer if ensemble_start_layer is not None else config.num_layers - 1

        # Define ensemble-specific transformer layers starting from the ensemble_start_layer
        self.ensemble_transformers = nn.ModuleList([
            nn.ModuleList(
                [self.transformer.h[i] for i in range(self.ensemble_start_layer, config.num_layers)]
            ) for _ in range(num_ensemble_heads)
        ])
        
        # Each ensemble head gets its own final Linear layer
        self.ensemble_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size) for _ in range(num_ensemble_heads)
        ])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Enable the model to return hidden states from all layers
        transformer_outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True,  # Enable hidden states for all layers
            **kwargs
        )

        # Retrieve the hidden states from the specified ensemble_start_layer
        hidden_states = transformer_outputs.hidden_states[self.ensemble_start_layer]

        # Ensure hidden_states is just the tensor, not a tuple
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Process the hidden states through the ensemble-specific transformers
        logits_list = []
        for ensemble_transformer, head in zip(self.ensemble_transformers, self.ensemble_heads):
            # Pass through each layer of the ensemble transformer individually
            ensemble_hidden_states = hidden_states
            for layer in ensemble_transformer:
                # Apply each transformer layer and ensure we're extracting only the hidden states
                ensemble_hidden_states = layer(ensemble_hidden_states)[0]  # Extract tensor from tuple

            # Pass the resulting hidden state to the ensemble head (Linear layer)
            logits = head(ensemble_hidden_states)
            logits_list.append(logits)

        if not self.training:
            # In evaluation mode, use all heads to compute mean and variance across the ensemble
            logits_mean = torch.stack(logits_list, dim=0).mean(dim=0)
            logits_var = torch.stack(logits_list, dim=0).var(dim=0)
            
            loss = None
            if labels is not None:
                shift_logits = logits_mean[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            return {"logits_mean": logits_mean, "logits_var": logits_var, "loss": loss}

        # In training mode, randomly select a subset of heads to train for each batch element
        batch_size = input_ids.size(0)
        logits_selected = []

        for i in range(batch_size):
            # Randomly select one or more heads for each example in the batch
            selected_heads_indices = random.sample(range(self.num_ensemble_heads), k=random.randint(1, self.num_ensemble_heads))
            selected_heads = [self.ensemble_heads[j] for j in selected_heads_indices]
            
            # Compute logits for selected heads
            selected_logits = [head(hidden_states[i:i+1]) for head in selected_heads]
            logits_selected.append(selected_logits)

        # Compute the loss using the selected heads' logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            total_loss = 0.0
            for i in range(batch_size):
                shift_logits = torch.stack(logits_selected[i], dim=0).mean(dim=0)[..., :-1, :].contiguous()
                shift_labels = labels[i:i+1, 1:].contiguous()
                total_loss += loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = total_loss / batch_size  # Average loss over batch

        return {"logits_mean": None, "logits_var": None, "loss": loss}

    # Override save_pretrained to include ensemble-specific layers
    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)  # Save the base GPT-Neo model and configuration
        torch.save(self.ensemble_transformers.state_dict(), f"{save_directory}/ensemble_transformers.pth")
        torch.save(self.ensemble_heads.state_dict(), f"{save_directory}/ensemble_heads.pth")

    # Override from_pretrained to load ensemble-specific layers
    @classmethod
    def from_finetuned(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(GPTNeoWithEnsembleHeads, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Load ensemble-specific layers
        ensemble_transformers_path = f"{pretrained_model_name_or_path}/ensemble_transformers.pth"
        ensemble_heads_path = f"{pretrained_model_name_or_path}/ensemble_heads.pth"

        model.ensemble_transformers.load_state_dict(torch.load(ensemble_transformers_path))
        model.ensemble_heads.load_state_dict(torch.load(ensemble_heads_path))

        return model


class GPTNeoWithLaplaceLastLayer(GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.laplace_covariance = None  # This will store the covariance matrix for the Laplace approximation
        self.laplace_method = "diag"  # Default is diagonal approximation

    def compute_last_layer_gn_approx(self, inputs, labels, laplace_method="diag"):
        """
        Computes the Gauss-Newton (GGN) approximation of the last layer's Hessian.
        """
        # Ensure model is in train mode for gradient tracking
        self.train()

        # Ensure that lm_head weights have requires_grad=True
        if not self.lm_head.weight.requires_grad:
            print("Warning: lm_head weight does not require grad. Setting requires_grad=True.")
            self.lm_head.weight.requires_grad_(True)

        # Forward pass to compute logits
        outputs = self(input_ids=inputs)
        logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute the cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Compute the Jacobian of the logits w.r.t. the model parameters (lm_head weight)
        # This is the first derivative of the loss w.r.t. the weights
        grads = torch.autograd.grad(loss, self.lm_head.weight, create_graph=True)[0]

        # Gauss-Newton approximation: J^T * J (Jacobian-transpose times Jacobian)
        if laplace_method == "diag":
            # Diagonal Gauss-Newton approximation: compute element-wise square of the Jacobian
            gn_diag_approx = grads.pow(2).sum(dim=0)
            return gn_diag_approx

        elif laplace_method == "full":
            # Full Gauss-Newton approximation: compute the full J^T * J matrix
            gn_full_approx = torch.einsum('bi,bj->ij', grads, grads)
            return gn_full_approx

        else:
            raise ValueError("Unsupported Laplace method: Choose 'diag' or 'full'.")


    def fit_laplace_approximation(self, dataloader, laplace_method="diag", laplace_weight_prior=1e-5, batch_size=1):
        """
        Computes the Laplace approximation of the last layer, using the Gauss-Newton approximation.
        We'll store the inverse of the diagonal or full approximation with L2 prior regularization.
        """
        self.laplace_method = laplace_method
        all_gn_results = []
    
        self.train()  # Enable gradients for the Hessian computation
    
        for batch_idx, batch in enumerate(dataloader):
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
    
            # Break the batch into smaller mini-batches to avoid memory overflow
            num_samples = inputs.size(0)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                input_batch = inputs[start:end]
                label_batch = labels[start:end]
    
                # Compute the GGN approximation for the mini-batch
                gn_result = self.compute_last_layer_gn_approx(input_batch, label_batch, laplace_method)
    
                # Detach the tensor and then convert to NumPy
                all_gn_results.append(gn_result.detach().cpu().numpy())  # Accumulate results as NumPy arrays
    
            torch.cuda.empty_cache()  # Clear cache after each mini-batch to free memory
    
        # Aggregate the GN approximations across all batches
        if laplace_method == "diag":
            # Mean of diagonal approximations over the data
            avg_gn_diag = np.mean(np.array(all_gn_results), axis=0)
            # Add L2 prior (weight decay) to the diagonal
            avg_gn_diag += laplace_weight_prior
            # Inverse of the approximation diagonal to get the covariance matrix
            self.laplace_covariance = torch.tensor(1.0 / avg_gn_diag).to(self.lm_head.weight.device)
    
        elif laplace_method == "full":
            # Mean of full approximations over the data
            avg_gn_full = np.mean(np.array(all_gn_results), axis=0)
    
            # Convert avg_gn_full back to a PyTorch tensor before using PyTorch operations
            avg_gn_full = torch.tensor(avg_gn_full).to(self.lm_head.weight.device)
    
            # Add L2 prior (weight decay) to the full approximation (by adding to identity matrix)
            avg_gn_full += laplace_weight_prior * torch.eye(avg_gn_full.shape[0], device=self.lm_head.weight.device)
    
            # Inverse of the approximation to get the covariance matrix
            self.laplace_covariance = torch.linalg.inv(avg_gn_full)
    
        else:
            raise ValueError("Unsupported Laplace method: Choose 'diag' or 'full'.")

    def save_laplace_covariance(self, file_path):
        """
        Save the Laplace covariance matrix to a file.

        Args:
            file_path (str): Path to the file where the covariance will be saved.
        """
        if self.laplace_covariance is None:
            raise ValueError("Laplace covariance has not been computed yet.")
        torch.save(self.laplace_covariance, file_path)
        print(f"Laplace covariance saved to {file_path}")

    def load_laplace_covariance(self, file_path):
        """
        Load the Laplace covariance matrix from a file.

        Args:
            file_path (str): Path to the file from which to load the covariance.
        """
        self.laplace_covariance = torch.load(file_path, map_location=self.lm_head.weight.device)
        print(f"Laplace covariance loaded from {file_path}")


def setup_model_and_trainer(cfg, datasets, tokenizer):
    model_config = cfg.model
    data_config = [d for d in cfg.datasets if d.name == cfg.stg.target_dataset_name][0]
    
    if model_config.var_method == 'dropout':
        # Load the pre-trained model and ensure it has dropout in all transformer layers
        model = GPTNeoForCausalLMWithDropout.from_pretrained(model_config['name']).to('cuda')
        # Apply the dropout rate to the entire model
        model.config.dropout_rate = model_config.dropout_rate    
        model.apply_dropout_rate(model.config.dropout_rate)
    elif model_config.var_method == 'ensemble':
        # Initialize the model with ensemble heads
        model = GPTNeoWithEnsembleHeads.from_pretrained(
            model_config['name'],
            num_ensemble_heads=model_config.n_ensemble_heads
        ).to('cuda')
    else:
        model = GPTNeoForCausalLM.from_pretrained(model_config['name']).to('cuda')

    
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        learning_rate=model_config.learning_rate,
        per_device_train_batch_size=model_config.train_batch_size,
        per_device_eval_batch_size=model_config.eval_batch_size,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        num_train_epochs=data_config.get('epochs', model_config.epochs),
        weight_decay=model_config.weight_decay,
        fp16=model_config.fp16,
        save_strategy=model_config.save_strategy,
        load_best_model_at_end=model_config.load_best_model_at_end,
        seed=0
    )

    if model_config.var_method == 'hfl':
        trainer = HFLTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            lam=model_config.hfl_lambda,
            compute_metrics=compute_metrics
        )
    elif model_config.var_method == 'hfl_oracle':
        # Set random seed for reproducibility of u_eval generation - use a separate seed from training
        u_seed = model_config.u_seed_init 
        log.info(f"Setting torch seed to {u_seed} for reproducible u_eval generation")
        torch.manual_seed(u_seed)
        
        # Calculate total number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

        # Generate random noise vector u_eval for each parameter with same shape
        u_eval = {}
        for name, param in model.named_parameters():
            # Generate standard normal noise and scale it by the target standard deviation
            u_eval[name] = (torch.randn_like(param) * target_std_dev).to(device)
            
        # Reset random seed to avoid affecting other random operations
        torch.manual_seed(torch.initial_seed())
        
        trainer = HFLOracleTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            lam=model_config.hfl_lambda,
            u_eval=u_eval,
            compute_metrics=compute_metrics
        )
    elif model_config.var_method == 'ensemble':
        trainer = EnsembleTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            compute_metrics=compute_metrics
        )

    return model, trainer


def apply_dropout_rate(model, dropout_rate):
    """
    Recursively apply the specified dropout rate to all dropout layers in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate  # Set the dropout rate
            log.info(f"Set dropout rate of layer {name} to {dropout_rate}")

# function to perform MC Dropout inference
def mc_dropout_predict(model, inputs, n_forward_passes=10):
    model.train()
    with torch.no_grad():
        outputs_list = []
        for _ in range(n_forward_passes):
            outputs = model(inputs)
            outputs_list.append(outputs.logits)
        stacked_outputs = torch.stack(outputs_list)
        # Compute mean and variance across the multiple forward passes
        mean_outputs = stacked_outputs.mean(dim=0)
        variance_outputs = stacked_outputs.var(dim=0)
    return mean_outputs, variance_outputs


