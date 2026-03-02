import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaTokenizer

from dataset_dmcl import CorpusDataset, ValidationQueriesDataset, QueryImageDataset

def extract_corpus_features(corpus_dataset: CorpusDataset, beit3_model: nn.Module, batch_size: int, device: torch.device):
    def corpus_collate_fn(batch):
        image_paths, images_tensors = zip(*batch)
        pixel_values = torch.stack(images_tensors)
        ids = [corpus_dataset.path_to_id_map[p] for p in image_paths]
        return torch.tensor(ids), pixel_values

    corpus_loader = DataLoader(
        dataset=corpus_dataset, batch_size=batch_size, num_workers=4,
        collate_fn=corpus_collate_fn, pin_memory=True
    )
    
    corpus_vectors_list, corpus_ids_list = [], []
    with torch.no_grad():
        for batch_ids, pixel_values in tqdm(corpus_loader, desc="Extracting Corpus Features (BEiT-3)"):
            pixel_values = pixel_values.to(device)
            batch_vectors, _ = beit3_model(image=pixel_values, only_infer=True)
            batch_vectors = F.normalize(batch_vectors, dim=-1)
            
            corpus_vectors_list.append(batch_vectors.cpu())
            corpus_ids_list.append(batch_ids.cpu())
            
    corpus_vectors = torch.cat(corpus_vectors_list)
    corpus_ids = torch.cat(corpus_ids_list)
    
    arg_ids = torch.argsort(corpus_ids)
    return corpus_ids[arg_ids].to(device), corpus_vectors[arg_ids].to(device)

def _create_or_load_generated_cache_beit3(model: nn.Module, preprocessor: callable, batch_size: int,
                                          queries_path: str, gen_image_dir: str, 
                                          num_eval_rounds: int, device: torch.device):
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    query_dataset = QueryImageDataset(queries=queries, gen_image_dir=gen_image_dir, num_rounds=num_eval_rounds, transform=preprocessor)
    query_dataset_loader = DataLoader(dataset=query_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    cache_data = {}
    model.eval()
    with torch.no_grad():
        for filenames, images in tqdm(query_dataset_loader, desc="Caching Generated Image Features"):
            images = images.to(device, non_blocking=True)
            gen_feats, _ = model(image=images, only_infer=True)
            gen_feats = F.normalize(gen_feats, dim=-1).detach().cpu()

            for i, filename in enumerate(filenames):
                cache_data[filename] = {"gen_feat": gen_feats[i]}
    return cache_data

def _calculate_fused_scores_beit3(method: str, text_features: torch.Tensor, gen_features: torch.Tensor, 
                                  corpus_features: torch.Tensor, dialog_length: int):
    corpus_features_T = corpus_features.T
    if method == 'text':
        return text_features @ corpus_features_T
    if method == 'image':
        return gen_features @ corpus_features_T
    if method == 'dar':
        text_scores = text_features @ corpus_features_T
        gen_scores = gen_features @ corpus_features_T
        w_text, w_img = (0.8, 0.2) if dialog_length < 2 else (0.5, 0.5)
        return w_text * text_scores + w_img * gen_scores
    if method == 'fused_feature':
        fused_features = F.normalize(gen_features + text_features, dim=-1)
        return fused_features @ corpus_features_T
    raise ValueError(f"Unknown fusion method: {method}")

def _calculate_ranks(ranked_indices, target_ids):
    ranks = []
    for i in range(ranked_indices.shape[0]):
        rank_tensor = (ranked_indices[i] == target_ids[i]).nonzero(as_tuple=True)[0]
        if rank_tensor.numel() > 0:
            ranks.append(rank_tensor.squeeze())
        else:
            ranks.append(torch.tensor(float('inf'), device=ranked_indices.device))
    return torch.stack(ranks)

def get_first_hitting_time(target_recall, num_rounds, hitting_recall=10):
    if len(target_recall) == 0: return torch.tensor([])
    target_recalls = target_recall.view(num_rounds, -1).T
    hits = (target_recalls < hitting_recall)
    final_hits = torch.inf * torch.ones(target_recalls.shape[0])
    hitting_times = []
    for ro_i in range(num_rounds):
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
    return torch.stack(hitting_times)

def cumulative_hits_per_round(target_recall, num_rounds, hitting_recall=10):
    if len(target_recall) == 0: return [0.0] * num_rounds
    ht_times = get_first_hitting_time(target_recall, num_rounds, hitting_recall)
    if ht_times.numel() == 0: return [0.0] * num_rounds
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times.shape[1])

def run_eval4_validation(beit3_model: nn.Module, tokenizer: XLMRobertaTokenizer, preprocessor: callable, 
                         hyper_params: dict, epoch: int, device: torch.device, is_ema: bool = False):
    print(f"\n--- Starting Validation (EMA={is_ema}) (R@10 Only) ---")
    beit3_model.eval()
    
    val_queries_dataset = ValidationQueriesDataset(
        queries_path=hyper_params['val_queries_path'], generated_image_dir=hyper_params['val_generated_image_dir']
    )
    corpus_val_dataset = CorpusDataset(
        json_file_path=hyper_params['val_corpus_json_path'], pil_transform=preprocessor
    )
    path_to_id_map = corpus_val_dataset.path_to_id_map
    
    corpus_ids, corpus_vectors = extract_corpus_features(corpus_val_dataset, beit3_model, hyper_params['batch_size'], device)
    num_eval_rounds = 11 
    gen_features_cache = _create_or_load_generated_cache_beit3(
        beit3_model, preprocessor, hyper_params['batch_size'], hyper_params['val_queries_path'], 
        hyper_params['val_generated_image_dir'], num_eval_rounds, device
    )

    experiments = {"BEiT3_Text_Only": "text", "BEiT3_Image_Only": "image", "BEiT3_DAR": "dar", "BEiT3_Fused_Feature": "fused_feature"}
    all_rounds_recalls = {name: [] for name in experiments.keys()}
    feature_dim = beit3_model.language_head.out_features
    zero_feature = torch.zeros((feature_dim,), device="cpu")

    for dl in range(num_eval_rounds):
        val_queries_dataset.set_dialog_length(dl)
        val_loader = DataLoader(val_queries_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        exp_recalls_per_round = {name: [] for name in experiments.keys()}
        
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val Round {dl}"):
            target_ids = torch.tensor([path_to_id_map.get(p, -1) for p in batch['target_path']], dtype=torch.long).unsqueeze(1).to(device)
            text_inputs = tokenizer(text=list(batch['text']), padding="longest", truncation=True, max_length=256, return_tensors="pt").to(device)
            
            _, text_features = beit3_model(text_description=text_inputs['input_ids'], padding_mask=1-text_inputs['attention_mask'], only_infer=True)
            
            gen_features_list = [gen_features_cache.get(os.path.basename(p), {}).get('gen_feat', zero_feature) for p in batch['gen_path']]
            gen_features = torch.stack(gen_features_list, dim=0).to(device, non_blocking=True)
            
            for name, method_type in experiments.items():
                total_scores = _calculate_fused_scores_beit3(method_type, text_features, gen_features, corpus_vectors, dl)
                arg_ranks = torch.argsort(total_scores, descending=True, dim=1).long()
                exp_recalls_per_round[name].append(_calculate_ranks(arg_ranks, target_ids))
        
        for name in experiments.keys():
            all_rounds_recalls[name].append(torch.cat(exp_recalls_per_round[name]) if exp_recalls_per_round[name] else torch.tensor([], dtype=torch.long))

    wandb_val_logs, epoch_results_for_excel = {}, {}
    model_prefix = "EMA" if is_ema else "Reg"

    for name, results_per_round in all_rounds_recalls.items():
        print(f"\n====== Experiment: '{name}' (R@10) ======")
        indep_r10_list, cumul_r10_list = [], []
        
        print(f"  --- Independent Per Round (R@10) ---")
        for dl, recalls in enumerate(results_per_round): 
            rate = (recalls < 10).sum().item() * 100 / len(recalls) if len(recalls) > 0 else 0.0
            indep_r10_list.append(rate)
            print(f"\tRound {dl}: {rate:.2f}%")
            wandb_val_logs[f"z_{model_prefix.lower()}_{name}_Indep_R{dl}_R@10"] = rate
        epoch_results_for_excel[f"{model_prefix}_{name}_Indep"] = indep_r10_list

        print(f"  --- Cumulative (R@10) ---")
        all_recalls_flat = torch.cat([r.cpu() for r in results_per_round if r.numel() > 0])
        cumulative_results = cumulative_hits_per_round(all_recalls_flat, num_rounds=num_eval_rounds, hitting_recall=10).tolist() if all_recalls_flat.numel() > 0 else [0.0] * num_eval_rounds
        
        for dl, rate in enumerate(cumulative_results):
            cumul_r10_list.append(rate)
            print(f"\tUp to Round {dl}: {rate:.2f}%")
            wandb_val_logs[f"z_{model_prefix.lower()}_{name}_Cumul_R{dl}_R@10"] = rate
        epoch_results_for_excel[f"{model_prefix}_{name}_Cumul"] = cumul_r10_list

    fused_metric = f"{model_prefix}_BEiT3_Fused_Feature_Indep"
    dar_metric = f"{model_prefix}_BEiT3_DAR_Indep"
    best_metric_for_checkpoint = epoch_results_for_excel.get(fused_metric, epoch_results_for_excel.get(dar_metric, [0.0]))[-1]

    return best_metric_for_checkpoint, epoch_results_for_excel, wandb_val_logs