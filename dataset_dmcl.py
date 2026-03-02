import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer

from dmcl_config import Config  # Centralized global configuration

class ComposedRetrievalDataset(Dataset):
    """
    Dataset for Training DMCL Framework.
    Yields triplets of (diffusion_image, target_image, accumulated_dialogue_caption).
    """
    def __init__(self, json_file_path: str, pil_transform: callable = None):
        super().__init__()
        self.json_file_path = Path(json_file_path)
        self.pil_transform = pil_transform
        
        # Read parameters directly from global Config
        self.dialogue_round = Config.dialogue_round
        self.use_random_rounds = Config.use_random_rounds
        self.diffusion_image_dir = Path(Config.train_diffusion_image_dir)
        
        self.diffusion_filename_prefix = "train-" 

        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data: List[Dict] = json.load(f)
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        item_info = self.data[index]
        target_path_str = item_info['img'] 
        dialog_list = item_info['dialog'] 
        
        # Determine the current round index for interaction
        if self.use_random_rounds:
            current_round_index = random.randint(0, self.dialogue_round)
        else:
            current_round_index = self.dialogue_round

        # VisDial format strictly: Accumulate dialogue history up to the current round
        caption = ", ".join(dialog_list[:current_round_index + 1])

        target_filename_stem = Path(target_path_str).stem
        
        target_image = Image.open(target_path_str).convert("RGB")
        if self.pil_transform:
            target_image = self.pil_transform(target_image)
            
        diffusion_image = self._load_diffusion_image(target_filename_stem, current_round_index)
        
        return diffusion_image, target_image, caption

    def _load_diffusion_image(self, target_filename_stem: str, round_idx: int):
        """
        Load the diffusion-generated visual proxy image based on target stem and round.
        """
        diffusion_filename = f"{self.diffusion_filename_prefix}{target_filename_stem}_{round_idx}.jpg"
        round_folder_name = f"round{round_idx}"
        diffusion_path = self.diffusion_image_dir / round_folder_name / diffusion_filename
        
        if not diffusion_path.exists():
            raise FileNotFoundError(f"Missing diffusion image: {diffusion_path}")
            
        image = Image.open(diffusion_path).convert("RGB")
        if self.pil_transform:
            image = self.pil_transform(image)
        return image
    
class CorpusDataset(Dataset):
    """
    Dataset for loading the target image corpus during validation.
    """
    def __init__(self, json_file_path: str, pil_transform: callable = None):
        super().__init__()
        self.json_file_path = Path(json_file_path)
        self.pil_transform = pil_transform

        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.image_paths: List[str] = json.load(f)

        self.path_to_id_map: Dict[str, int] = {path: i for i, path in enumerate(self.image_paths)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.pil_transform:
            image = self.pil_transform(image)
            
        return image_path, image
    
class ValidationQueriesDataset(Dataset):
    """
    Dataset for loading validation queries across different dialogue rounds.
    """
    def __init__(self, queries_path: str, generated_image_dir: str):
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
            
        self.generated_image_dir = Path(generated_image_dir)
        self.dialog_length = 0 
        self.sep_token = ", "

    def __len__(self) -> int:
        return len(self.queries)

    def set_dialog_length(self, dialog_length: int):
        self.dialog_length = dialog_length

    def __getitem__(self, i: int) -> Dict:
        target_path = self.queries[i]['img']
        
        # Accumulate dialogue history for VisDial format
        text = self.sep_token.join(self.queries[i]['dialog'][:self.dialog_length + 1])

        gen_image_filename = f"{i}_{self.dialog_length}.jpg"
        gen_image_path = (self.generated_image_dir / gen_image_filename).as_posix()

        return {
            'query_idx': i,       
            'text': text,         
            'target_path': target_path, 
            'gen_path': gen_image_path  
        }
    
class QueryImageDataset(Dataset):
    """
    Dataset for caching the features of diffusion-generated query images.
    """
    def __init__(self, queries: List[Dict], gen_image_dir: str, num_rounds: int, transform: callable = None):
        self.samples = []
        self.transform = transform
        gen_dir = Path(gen_image_dir)
        
        for query_idx in range(len(queries)):
            for round_idx in range(num_rounds):
                filename = f"{query_idx}_{round_idx}.jpg"
                filepath = gen_dir / filename
                
                if filepath.exists():
                    self.samples.append((filename, str(filepath)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        filename, filepath = self.samples[idx]
       
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return filename, image
    
def beit3_collate_fn(batch: list, tokenizer: XLMRobertaTokenizer):

    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None, None

    diff_pixel_values, target_pixel_values, captions = zip(*batch)

    diff_pixel_values = torch.stack(diff_pixel_values)
    target_pixel_values = torch.stack(target_pixel_values)

    text_inputs = tokenizer(
            text=list(captions), 
            padding="longest",  
            truncation=True,       
            max_length=256,        
            return_tensors="pt"
        )

    return diff_pixel_values, target_pixel_values, text_inputs