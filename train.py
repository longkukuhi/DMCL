import sys
import os

# Ensure 'beit3' folder is in the search path
current_dir = os.path.dirname(os.path.abspath(__file__))
beit3_dir = os.path.join(current_dir, 'beit3')
if beit3_dir not in sys.path:
    sys.path.insert(0, beit3_dir)

import wandb
import json
from typing import List, Dict, Tuple
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from torchvision import transforms 
from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, get_num_layer_for_vit
from dataset_dmcl import ComposedRetrievalDataset, beit3_collate_fn
from train_utils import update_train_running_results, set_train_bar_description, build_beit3_transform, save_checkpoint
import utils
import modeling_finetune
from modeling_utils import _get_large_config, _get_base_config
from eval_utils import run_eval4_validation
from dmcl_config import Config

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class BEiT3ForRetrievalWrapper(modeling_finetune.BEiT3ForRetrieval):

    def __init__(self, args, **kwargs):
        # Call parent init to retain all original logic
        super().__init__(args, **kwargs)
        
        # Extension: Initialize learnable weights
        self.use_learnable_weights = getattr(args, 'use_learnable_weights', False)
        
        if self.use_learnable_weights:
            self.loss_param_dict = nn.ParameterDict()
            
            # Get configuration
            components = getattr(args, 'loss_components', [])
            weights = getattr(args, 'loss_weights', [])

            # Strict check for configuration consistency
            if len(components) != len(weights):
                error_msg = (
                    f"\n[Fatal Error] Severe configuration error!\n"
                    f"loss_components (len={len(components)}): {components}\n"
                    f"loss_weights    (len={len(weights)}): {weights}\n"
                    f"Reason: Lengths must match strictly. Please check 'beit3_config_js.py'."
                )
                raise ValueError(error_msg)
                
            if len(components) == 0:
                raise ValueError("\n[Fatal Error] loss_components is empty! At least one loss function must be specified.")

            print(f"\n[BEiT3Wrapper] Initializing Learnable Weights:")
            print(f"{'Component':<20} | {'Target Weight':<15} | {'Init Param (Log)':<20}")
            print("-" * 60)

            for name, target_val in zip(components, weights):
                target_val = float(target_val)
                
                # Safety check: weights cannot be <= 0 for Softplus
                if target_val <= 1e-6:
                    print(f"[Warning] Target weight for '{name}' is {target_val} (close to 0).")
                    # Initialize to a small negative number (e.g., -5.0), Softplus(-5.0) ≈ 0.0067
                    init_param_val = -5.0 
                else:
                    try:
                        # Inverse Softplus initialization: log(exp(target) - 1)
                        init_param_val = np.log(np.exp(target_val) - 1)
                    except Exception as e:
                         raise ValueError(f"Math error calculating init param (Target={target_val}): {e}")
                
                # Create learnable parameter
                self.loss_param_dict[name] = nn.Parameter(torch.tensor(init_param_val, dtype=torch.float32))
                
                print(f"{name:<20} | {target_val:<15.4f} | {init_param_val:<20.4f}")
            
            print(f"{'-'*60}\n")

def compute_hnm_loss(logits, ground_truth, topk, margin, temperature=1.0):
    """
    Compute Hard Negative Mining (HNM) Loss.
    """
    logits = logits.float()
    batch_size = logits.size(0)
    
    # 1. Get scores for positive pairs (N, 1)
    pos_scores = logits.gather(1, ground_truth.view(-1, 1))

    # 2. Mask positive pairs to find negatives
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, ground_truth.view(-1, 1), True)
    neg_logits = logits.clone()
    neg_logits.masked_fill_(mask, float('-inf'))

    # 3. Find Top-K hard negatives
    actual_k = min(topk, batch_size - 1)
    if actual_k <= 0:
        return torch.tensor(0.0, device=logits.device), 0.0, 0.0
        
    # hard_neg_scores: (N, K)
    hard_neg_scores, _ = torch.topk(neg_logits, k=actual_k, dim=1)

    # 4. Compute Loss
    # delta = s_neg - s_pos + m
    delta = (hard_neg_scores - pos_scores + margin) / temperature
    loss_matrix = F.softplus(delta) 
    loss = loss_matrix.mean()
    
    # Statistics
    mean_pos_score = pos_scores.detach().mean().item()
    mean_neg_score = hard_neg_scores.detach().mean().item()
    
    return loss, mean_pos_score, mean_neg_score

def compute_js_divergence(logits_p, logits_q, temperature=1.0):
    """
    Compute Jensen-Shannon Divergence between two Logits distributions.
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
    """
    # 1. Temperature Scaling
    logits_p = logits_p / temperature
    logits_q = logits_q / temperature
    
    # 2. Compute probabilities (Softmax)
    p_prob = F.softmax(logits_p, dim=-1)
    q_prob = F.softmax(logits_q, dim=-1)
    
    # 3. Compute mean distribution M
    m_prob = 0.5 * (p_prob + q_prob)
    
    # 4. Numerical stability (prevent log(0))
    eps = 1e-8
    p_prob = torch.clamp(p_prob, min=eps)
    q_prob = torch.clamp(q_prob, min=eps)
    m_prob = torch.clamp(m_prob, min=eps)
    
    # 5. Compute KL Divergence: KL(P||M) = sum(P * (log P - log M))
    kl_p_m = (p_prob * (p_prob.log() - m_prob.log())).sum(dim=-1).mean()
    kl_q_m = (q_prob * (q_prob.log() - m_prob.log())).sum(dim=-1).mean()
    
    # 6. Compute JS Divergence
    js_divergence = 0.5 * kl_p_m + 0.5 * kl_q_m
    
    return js_divergence

def train_beit3_finetune(**hyper_params):

    train_json_path = hyper_params['train_json_path']
    num_epochs = hyper_params['num_epochs']
    batch_size = hyper_params['batch_size']
    validation_frequency = hyper_params['validation_frequency']
    save_training = hyper_params['save_training']
    resume_from = hyper_params.get('resume_from')
    drop_path_rate = hyper_params.get('drop_path', 0.1)

    loss_components = hyper_params['loss_components']

    args = _get_base_config(
        img_size=hyper_params['input_size'], 
        vocab_size=64010, 
        drop_path_rate=drop_path_rate,
        checkpoint_activations=True  
    )

    args.use_learnable_weights = hyper_params.get('use_learnable_weights', False)
    args.loss_components = hyper_params.get('loss_components', [])
    args.loss_weights = hyper_params.get('loss_weights', [])

    beit3_model = BEiT3ForRetrievalWrapper(args=args)
    beit3_model.to(device)

    training_path = Path(hyper_params['log_base_dir']) / hyper_params['experiment_name']
    training_path.mkdir(exist_ok=True, parents=True)

    hnm_stats_excel_path = training_path / 'hnm_detailed_statistics.xlsx'

    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(hyper_params, file, sort_keys=True, indent=4)
    train_log_path = training_path / 'train_metrics.csv'

    if resume_from and train_log_path.exists():
        training_log_frame = pd.read_csv(train_log_path)
    else:
        training_log_frame = pd.DataFrame()

    val_excel_path = training_path / 'validation_summary.xlsx'
    validation_dataframes = {}
    if resume_from and val_excel_path.exists():
        validation_dataframes = pd.read_excel(str(val_excel_path), sheet_name=None, index_col=0)

    checkpoint_path = hyper_params['beit3_checkpoint_path']
    utils.load_model_and_may_interpolate(
        ckpt_path=checkpoint_path, 
        model=beit3_model, 
        model_key='model', 
        model_prefix=''
    )  

    beit3_model.to(device).train()

    tokenizer_path = hyper_params['beit3_tokenizer_path']
    tokenizer = XLMRobertaTokenizer(tokenizer_path)

    transform_train = build_beit3_transform(is_train=False, config=hyper_params)
    transform_val = build_beit3_transform(is_train=False, config=hyper_params)
    
    train_dataset = ComposedRetrievalDataset(
        json_file_path=train_json_path, 
        pil_transform=transform_train
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(beit3_collate_fn, tokenizer=tokenizer),
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )

    model_ema = None
    if hyper_params.get('model_ema', False):
        model_ema = ModelEma(
            beit3_model,
            decay=hyper_params['model_ema_decay'],
            device='cpu' if False else '',
            resume=''
        )

    params_to_optimize = []
    beit3_lr = hyper_params['beit3_lr']
    layer_decay = hyper_params['layer_decay']
    num_layers = beit3_model.get_num_layers()
    if layer_decay < 1.0:
        lr_scales = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        layer_decay_assigner = LayerDecayValueAssigner(lr_scales)

    beit3_param_groups = get_parameter_groups(
        beit3_model, hyper_params['weight_decay'], beit3_model.no_weight_decay(),
        get_num_layer=layer_decay_assigner.get_layer_id,
        get_layer_scale=layer_decay_assigner.get_scale
    )

    for group in beit3_param_groups:
        group['lr'] = beit3_lr * group.get('lr_scale', 1.0)
    params_to_optimize.extend(beit3_param_groups)

    optimizer = optim.AdamW(params_to_optimize, eps=1e-8, betas=(0.9, 0.999))
    num_training_steps = (len(train_loader) * num_epochs) // hyper_params['update_freq']
    num_warmup_steps = (len(train_loader) * hyper_params.get('warmup_epochs', 2)) // hyper_params['update_freq']

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0
    best_recall_at_10 = 0.0

    if resume_from and Path(resume_from).exists():
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cpu')
        
        if 'model' in checkpoint:
            beit3_model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            beit3_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Neither 'model' nor 'model_state_dict' found in checkpoint.")

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored.")
        else:
            print("Warning: Scheduler state not found, starting fresh.")
            
        if model_ema is not None and 'model_ema_state_dict' in checkpoint:
            model_ema.ema.load_state_dict(checkpoint['model_ema_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1 
        best_recall_at_10 = checkpoint.get('best_recall_at_10', 0.0)      
        print(f"Resumed successfully! Starting from Epoch {start_epoch}. Current Best R@10: {best_recall_at_10:.2f}%")

    print('Training loop started')
    for epoch in range(start_epoch, num_epochs):
        beit3_model.train()
        epoch_batch_details = []

        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0.0}
        for loss_name in hyper_params['loss_components']:
            train_running_results[f'accumulated_loss_{loss_name}'] = 0.0
        train_bar = tqdm(train_loader, ncols=150)
        
        for i, (diff_pixel_values, target_pixel_values, text_inputs) in enumerate(train_bar):

            current_batch_stats = {'batch_idx': i}
            images_in_batch = diff_pixel_values.size(0)
            step = len(train_loader) * epoch + i
            current_step_logs = {
                "train/step": step,
                "train/epoch": epoch + i / len(train_loader),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/logit_scale": beit3_model.logit_scale.exp().item() 
            }
            diff_pixel_values = diff_pixel_values.to(device)
            target_pixel_values = target_pixel_values.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.cuda.amp.autocast():
                diffusion_features, _ = beit3_model(image=diff_pixel_values, only_infer=True)
                target_features, _ = beit3_model(image=target_pixel_values, only_infer=True)                   
                padding_mask = 1 - text_inputs['attention_mask']
                _, text_features = beit3_model(
                    text_description=text_inputs['input_ids'],
                    padding_mask=padding_mask,
                    only_infer=True
                )

                logit_scale_exp = beit3_model.logit_scale.exp()
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss_components_to_use = set(hyper_params['loss_components'])
                loss_weights = dict(zip(hyper_params['loss_components'], hyper_params['loss_weights']))

                diffusion_features_norm = F.normalize(diffusion_features, dim=-1, eps=1e-6)
                target_features_norm = F.normalize(target_features, dim=-1, eps=1e-6)
                text_features_norm = F.normalize(text_features, dim=-1, eps=1e-6)
                
                fused_features_norm = None
                if 'fused_tgt' in loss_components_to_use:
                    predicted_features = diffusion_features + text_features
                    fused_features_norm = F.normalize(predicted_features, dim=-1, eps=1e-6)

                use_hnm = hyper_params['use_hnm']
                hnm_weight = hyper_params['hnm_weight']
                hnm_topk = hyper_params['hnm_topk']
                hnm_margin = hyper_params['hnm_margin']
                hnm_temp = hyper_params['hnm_temp']

                all_losses = {}     
                pure_losses = {}    

                if 'diff_tgt' in loss_components_to_use:
                    sim_diff_to_tgt = diffusion_features_norm @ target_features_norm.T
                    sim_tgt_to_diff = sim_diff_to_tgt.T 

                    logits_diff_to_tgt = logit_scale_exp * sim_diff_to_tgt
                    logits_tgt_to_diff = logit_scale_exp * sim_tgt_to_diff
                    
                    loss_diff_tgt = criterion(logits_diff_to_tgt, ground_truth)
                    loss_tgt_diff = criterion(logits_tgt_to_diff, ground_truth)
                    loss_diff_tgt_total = (loss_diff_tgt + loss_tgt_diff) / 2

                    pure_losses['diff_tgt'] = loss_diff_tgt_total.detach().item()

                    if use_hnm:
                        hnm_diff, pos_diff, neg_diff = compute_hnm_loss(sim_diff_to_tgt, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_tgt, pos_tgt, neg_tgt = compute_hnm_loss(sim_tgt_to_diff, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_loss_val = (hnm_diff + hnm_tgt) / 2
                        
                        key_log = 'accumulated_loss_hnm_diff'
                        if key_log not in train_running_results: train_running_results[key_log] = 0.0
                        train_running_results[key_log] += hnm_loss_val.detach().item() * images_in_batch
                        
                        loss_diff_tgt_total = loss_diff_tgt_total + hnm_weight * hnm_loss_val

                        avg_pos = (pos_diff + pos_tgt) / 2
                        avg_neg = (neg_diff + neg_tgt) / 2
                        diff_val = avg_pos - avg_neg
                        
                        current_step_logs.update({
                            "hnm_stats/diff_pos": avg_pos,
                            "hnm_stats/diff_neg": avg_neg,
                            "hnm_stats/diff_diff": diff_val,
                            "train/loss_hnm_diff": hnm_loss_val.detach().item()
                        })
                        
                        current_batch_stats['diff_pos'] = avg_pos
                        current_batch_stats['diff_neg'] = avg_neg
                        current_batch_stats['diff_diff'] = diff_val

                    all_losses['diff_tgt'] = loss_diff_tgt_total

                if 'text_tgt' in loss_components_to_use:
                    sim_text_to_tgt = text_features_norm @ target_features_norm.T
                    sim_tgt_to_text = sim_text_to_tgt.T

                    logits_text_to_tgt = logit_scale_exp * sim_text_to_tgt
                    logits_tgt_to_text = logit_scale_exp * sim_tgt_to_text

                    loss_text_tgt = criterion(logits_text_to_tgt, ground_truth)
                    loss_tgt_text = criterion(logits_tgt_to_text, ground_truth)
                    loss_text_tgt_total = (loss_text_tgt + loss_tgt_text) / 2

                    pure_losses['text_tgt'] = loss_text_tgt_total.detach().item()

                    if use_hnm:
                        hnm_text, pos_text, neg_text = compute_hnm_loss(
                                sim_text_to_tgt, ground_truth, hnm_topk, hnm_margin, hnm_temp
                            )
                        hnm_tgt_text, pos_tgt_text, neg_tgt_text = compute_hnm_loss(
                                sim_tgt_to_text, ground_truth, hnm_topk, hnm_margin, hnm_temp
                            )
                        hnm_loss_val_text = (hnm_text + hnm_tgt_text) / 2
                        
                        key_log = 'accumulated_loss_hnm_text'
                        if key_log not in train_running_results: train_running_results[key_log] = 0.0
                        train_running_results[key_log] += hnm_loss_val_text.detach().item() * images_in_batch

                        loss_text_tgt_total = loss_text_tgt_total + hnm_weight * hnm_loss_val_text

                        avg_pos_text = (pos_text + pos_tgt_text) / 2
                        avg_neg_text = (neg_text + neg_tgt_text) / 2
                        diff_val_text = avg_pos_text - avg_neg_text
                        
                        current_step_logs.update({
                            "hnm_stats/text_pos": avg_pos_text,
                            "hnm_stats/text_neg": avg_neg_text,
                            "hnm_stats/text_diff": diff_val_text,
                            "train/loss_hnm_text": hnm_loss_val_text.detach().item()
                        })
                        
                        current_batch_stats['text_pos'] = avg_pos_text
                        current_batch_stats['text_neg'] = avg_neg_text
                        current_batch_stats['text_diff'] = diff_val_text
                        
                    all_losses['text_tgt'] = loss_text_tgt_total

                if 'fused_tgt' in loss_components_to_use:
                    sim_fused_to_tgt = fused_features_norm @ target_features_norm.T
                    sim_tgt_to_fused = sim_fused_to_tgt.T

                    logits_fused_to_tgt = logit_scale_exp * sim_fused_to_tgt
                    logits_tgt_to_fused = logit_scale_exp * sim_tgt_to_fused

                    loss_fused_to_tgt = criterion(logits_fused_to_tgt, ground_truth)
                    loss_tgt_to_fused = criterion(logits_tgt_to_fused, ground_truth)
                    loss_fused_tgt_total = (loss_fused_to_tgt + loss_tgt_to_fused) / 2

                    pure_losses['fused_tgt'] = loss_fused_tgt_total.detach().item()
                    
                    if use_hnm:
                        hnm_fused, pos_fused, neg_fused = compute_hnm_loss(sim_fused_to_tgt, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_tgt_fused, pos_tgt_fused, neg_tgt_fused = compute_hnm_loss(sim_tgt_to_fused, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_loss_val_fused = (hnm_fused + hnm_tgt_fused) / 2
                        
                        key_log = 'accumulated_loss_hnm_fused'
                        if key_log not in train_running_results: train_running_results[key_log] = 0.0
                        train_running_results[key_log] += hnm_loss_val_fused.detach().item() * images_in_batch

                        loss_fused_tgt_total = loss_fused_tgt_total + hnm_weight * hnm_loss_val_fused

                        avg_pos_fused = (pos_fused + pos_tgt_fused) / 2
                        avg_neg_fused = (neg_fused + neg_tgt_fused) / 2
                        diff_val_fused = avg_pos_fused - avg_neg_fused
                        
                        current_step_logs.update({
                            "hnm_stats/fused_pos": avg_pos_fused,
                            "hnm_stats/fused_neg": avg_neg_fused,
                            "hnm_stats/fused_diff": diff_val_fused,
                            "train/loss_hnm_fused": hnm_loss_val_fused.detach().item()
                        })

                        current_batch_stats['fused_pos'] = avg_pos_fused
                        current_batch_stats['fused_neg'] = avg_neg_fused
                        current_batch_stats['fused_diff'] = diff_val_fused

                    all_losses['fused_tgt'] = loss_fused_tgt_total
                    
                if 'diff_text' in loss_components_to_use:
                    logits_text_to_diff = logit_scale_exp * text_features_norm @ diffusion_features_norm.T
                    logits_diff_to_text = logit_scale_exp * diffusion_features_norm @ text_features_norm.T
                    loss_text_diff = criterion(logits_text_to_diff, ground_truth)
                    loss_diff_text = criterion(logits_diff_to_text, ground_truth)
                    loss_diff_text_total = (loss_text_diff + loss_diff_text) / 2

                    all_losses['diff_text'] = loss_diff_text_total
                    pure_losses['diff_text'] = loss_diff_text_total.detach().item()

                if 'dist_agreement' in loss_components_to_use:
                    logits_T = logit_scale_exp * (text_features_norm @ target_features_norm.T)
                    logits_D = logit_scale_exp * (diffusion_features_norm @ target_features_norm.T)
                    
                    dist_temp = hyper_params.get('dist_loss_temp', 1.0)
                    
                    js_loss_val = compute_js_divergence(logits_T, logits_D, temperature=dist_temp)
                    
                    all_losses['dist_agreement'] = js_loss_val
                    pure_losses['dist_agreement'] = js_loss_val.detach().item()
                    
                if not all_losses:
                    raise ValueError("Config Error: 'loss_components' is empty or invalid.")
                
                total_loss_weighted = 0.0
                use_learnable = getattr(beit3_model, 'use_learnable_weights', False)
                total_weight_sum = 0.0
                for name, loss_value in all_losses.items():
                    if use_learnable:
                        if name in beit3_model.loss_param_dict:
                            raw_param = beit3_model.loss_param_dict[name]
                            weight = F.softplus(raw_param)
                        else:
                            raise ValueError(f"[Error] Loss '{name}' has no corresponding learnable parameter.")
                    else:
                        weight = loss_weights.get(name, 1.0)

                    weighted_component = loss_value * weight
                    total_loss_weighted += weighted_component

                    total_weight_sum += weight 
                    weight_val = weight.item() if isinstance(weight, torch.Tensor) else weight

                    current_step_logs[f"train/weight_{name}"] = weight_val
                    current_step_logs[f"train/loss_component_{name}_weighted"] = weighted_component.item()
                    current_step_logs[f"train/loss_component_{name}_pure"] = pure_losses.get(name, 0.0)

                    key = f'accumulated_loss_{name}'
                    if key in train_running_results:
                        train_running_results[key] += loss_value.item() * images_in_batch

                loss = total_loss_weighted / (total_weight_sum + 1e-8)   
                unscaled_loss = loss 

                current_step_logs["train/loss_total"] = unscaled_loss.detach().item()
                if isinstance(total_weight_sum, torch.Tensor):
                    current_step_logs["train/total_weight_sum"] = total_weight_sum.item()
                else:
                    current_step_logs["train/total_weight_sum"] = total_weight_sum
                                      
                wandb.log(current_step_logs)

            loss = loss / hyper_params['update_freq']

            scaler.scale(loss).backward()
            if (i + 1) % hyper_params['update_freq'] == 0:
                max_norm = hyper_params['clip_grad']
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(beit3_model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                if model_ema is not None:
                    model_ema.update(beit3_model)
                optimizer.zero_grad() 
                scheduler.step()

            update_train_running_results(train_running_results, unscaled_loss.detach(), images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            if use_hnm:
                epoch_batch_details.append(current_batch_stats)
        
        epoch_metrics_data = {'epoch': epoch}
        total_images = train_running_results['images_in_epoch']

        train_epoch_loss = float(train_running_results['accumulated_train_loss'] / total_images)
        epoch_metrics_data['train_epoch_loss'] = train_epoch_loss
        wandb.log({"train/epoch_avg_loss": train_epoch_loss, "epoch": epoch})

        for loss_name in hyper_params['loss_components']:
            key = f'accumulated_loss_{loss_name}'
            if key in train_running_results and total_images > 0:
                epoch_loss_component = train_running_results[key] / total_images
                epoch_metrics_data[f'loss_{loss_name}'] = epoch_loss_component
                
        hnm_suffixes = ['diff', 'text', 'fused'] 
        for suffix in hnm_suffixes:
            key = f'accumulated_loss_hnm_{suffix}' 
            if key in train_running_results and total_images > 0:
                epoch_loss_component = train_running_results[key] / total_images
                epoch_metrics_data[f'loss_hnm_{suffix}'] = epoch_loss_component

        training_log_frame = pd.concat([
            training_log_frame,
            pd.DataFrame(data=epoch_metrics_data, index=[0])
        ])

        training_log_frame = training_log_frame.drop_duplicates(subset=['epoch'], keep='last')
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if use_hnm and len(epoch_batch_details) > 0:
            df_epoch_hnm = pd.DataFrame(epoch_batch_details)
            sheet_name = f'Epoch_{epoch}'
            
            mode = 'a' if hnm_stats_excel_path.exists() else 'w'
            if_sheet_exists = 'replace' if mode == 'a' else None
            
            try:
                with pd.ExcelWriter(hnm_stats_excel_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                    df_epoch_hnm.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                print(f"[Warning] Failed to save HNM statistics: {e}")

        if epoch % validation_frequency == 0:
            beit3_model.eval()
            all_epoch_metrics = {}
            
            with torch.no_grad():
                current_r10, regular_metrics_log, wandb_reg_logs = run_eval4_validation(
                    beit3_model=beit3_model,
                    tokenizer=tokenizer,
                    preprocessor=transform_val, 
                    hyper_params=hyper_params,
                    epoch=epoch,
                    device=device,
                    is_ema=False
                )

            wandb_reg_logs['epoch'] = epoch
            wandb.log(wandb_reg_logs)
        
            all_epoch_metrics.update(regular_metrics_log)
            print(f"Base Model R@10 (Round 10, DAR): {current_r10:.2f}%")
            current_best_metric = current_r10

            if model_ema is not None:
                print("\n--- Validating EMA Model ---")
                model_ema.ema.eval()

                with torch.no_grad():
                    ema_r10, ema_metrics_log, wandb_ema_logs = run_eval4_validation(
                        beit3_model=model_ema.ema,
                        tokenizer=tokenizer,
                        preprocessor=transform_val, 
                        hyper_params=hyper_params,
                        epoch=epoch,
                        device=device,
                        is_ema=True
                    )
                wandb_ema_logs['epoch'] = epoch
                wandb.log(wandb_ema_logs)
            
                all_epoch_metrics.update(ema_metrics_log)
                print(f"EMA Model R@10 (Round 10, DAR): {ema_r10:.2f}%")
                current_best_metric = ema_r10 
            col_names = [f'Round {i}' for i in range(11)]
            for sheet_name, new_data_list in all_epoch_metrics.items():
                
                new_row_df = pd.DataFrame([new_data_list], columns=col_names, index=[epoch])
                new_row_df.index.name = "Epoch"
                
                if sheet_name in validation_dataframes:
                    existing_df = validation_dataframes[sheet_name]
                    if epoch in existing_df.index:
                        existing_df.loc[epoch] = new_data_list
                    else:
                        validation_dataframes[sheet_name] = pd.concat([existing_df, new_row_df])
                else:
                    validation_dataframes[sheet_name] = new_row_df
            with pd.ExcelWriter(str(val_excel_path), engine='openpyxl') as writer:
                    for sheet_name, df in validation_dataframes.items():
                        df.to_excel(writer, sheet_name=sheet_name)

            if save_training:
                if current_best_metric > best_recall_at_10:
                    best_recall_at_10 = current_best_metric
                    print(f"New best model found (R@10: {best_recall_at_10:.2f}%), saving 'best' checkpoint...")
                    save_checkpoint('best', epoch, beit3_model, optimizer, scaler, best_recall_at_10, training_path,scheduler,  model_ema)

                print(f"Saving checkpoint for Epoch {epoch}... (Best R@10: {best_recall_at_10:.2f}%)")
                save_checkpoint(epoch, epoch, beit3_model, optimizer, scaler, best_recall_at_10, training_path, scheduler, model_ema,)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=Config.experiment_name, help="Name of the experiment and log folder")
    parser.add_argument("--log_base_dir", type=str, default="experiments", help="Root directory for all experiment logs")
    
    # --- Data and Model Path Parameters ---
    parser.add_argument("--dialogue_round", type=int, default=Config.dialogue_round,
                        help="Number of dialogue rounds (0-10)")
    parser.add_argument('--use_random_rounds', action='store_true',default=Config.use_random_rounds, help="Enable random dialogue rounds strategy (R3)")
    parser.add_argument("--train_json_path", type=str, default=Config.train_json_path)
    parser.add_argument("--val_corpus_json_path", type=str, default=Config.val_corpus_json_path, help="Path to large-scale validation corpus")
    parser.add_argument("--beit3_checkpoint_path", type=str, default=Config.beit3_checkpoint_path, help="Path to BEiT-3 pretrained weights (.pth)")
    parser.add_argument("--beit3_tokenizer_path", type=str, default=Config.beit3_tokenizer_path, help="Path to BEiT-3 tokenizer model (.spm)")
    parser.add_argument("--val_queries_path", type=str, default=Config.val_queries_path)
    parser.add_argument("--val_generated_image_dir", type=str, default=Config.val_generated_image_dir)
    parser.add_argument("--train_diffusion_image_dir", type=str, 
                        default=Config.train_diffusion_image_dir,
                        help="Folder path for training diffusion images")

    parser.add_argument("--num_epochs", default=Config.num_epochs, type=int)
    parser.add_argument("--beit3_lr", default=Config.beit3_lr, type=float)
    parser.add_argument("--weight_decay", default=Config.weight_decay, type=float, help="Weight decay (not applied to bias/LayerNorm)")
    parser.add_argument("--warmup_epochs", default=Config.warmup_epochs, type=int, help="Number of warmup epochs")
    parser.add_argument("--layer_decay", default=Config.layer_decay, type=float, help="Layer-wise learning rate decay (enabled if < 1.0)")
    parser.add_argument("--drop_path", default=Config.drop_path, type=float, help="DropPath rate for BEiT-3")
    parser.add_argument("--batch_size", default=Config.batch_size, type=int)
    parser.add_argument("--update_freq", default=Config.update_freq, type=int)
    parser.add_argument("--clip_grad", default=Config.clip_grad, type=float)
    parser.add_argument("--model_ema", action='store_true', default=Config.model_ema)
    parser.add_argument("--model_ema_decay", type=float, default=Config.model_ema_decay)
    parser.add_argument("--validation_frequency", default=Config.validation_frequency, type=int)

    parser.add_argument("--resume_from", type=str, default=Config.resume_from, help="Resume training from specified checkpoint")
    parser.add_argument("--save_training", action='store_true', default=Config.save_training)
    
    # --- Other Configs ---
    parser.add_argument("--input_size", type=int, default=Config.input_size)

    parser.add_argument("--loss_components", nargs='+', default=Config.loss_components, 
                        help="List of loss components to use (e.g., diff_tgt text_tgt)")
    parser.add_argument("--loss_weights", nargs='+', type=float, default=Config.loss_weights,
                        help="Weights corresponding to loss_components")
    
    parser.add_argument("--use_hnm", action='store_true', default=Config.use_hnm, help="Enable HNM regularization")
    parser.add_argument("--hnm_weight", type=float, default=Config.hnm_weight, help="HNM Loss weight")
    parser.add_argument("--hnm_topk", type=int, default=Config.hnm_topk, help="HNM Top-K negatives count")
    parser.add_argument("--hnm_margin", type=float, default=Config.hnm_margin, help="HNM Margin")
    parser.add_argument("--hnm_temp", type=float, default=Config.hnm_temp, help="HNM Temperature")

    parser.add_argument("--wandb_project", type=str, default=Config.wandb_project, help="WandB Project Name")
    parser.add_argument("--wandb_entity", type=str, default=Config.wandb_entity, help="WandB Username/Team")
    parser.add_argument("--wandb_mode", type=str, default=Config.wandb_mode, 
                        choices=['online', 'offline', 'disabled'], 
                        help="WandB mode: online, offline, disabled")
    
    parser.add_argument("--use_learnable_weights", action='store_true', 
                        default=getattr(Config, 'use_learnable_weights', False),
                        help="Enable learnable loss weights")
    
    parser.add_argument("--dist_loss_temp", type=float, 
                        default=getattr(Config, 'dist_loss_temp', 1.0), 
                        help="Temperature for JS Divergence Loss")

    args = parser.parse_args()

    training_hyper_params = vars(args)
    if len(training_hyper_params['loss_components']) != len(training_hyper_params['loss_weights']):
        raise ValueError("The number of loss_components and loss_weights must match!")

    run_id = args.experiment_name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        id=run_id,         
        resume="allow",    
        config=training_hyper_params,
        mode=args.wandb_mode, 
        settings=wandb.Settings(start_method="fork") 
    )
    # Start Training
    train_beit3_finetune(**training_hyper_params)