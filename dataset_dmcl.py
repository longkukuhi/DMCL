import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import torchvision.transforms.functional as F
import torch
import random
from dmcl_config import Config

class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """
    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')
        
class ComposedRetrievalDataset(Dataset):

    def __init__(self, json_file_path: str, pil_transform: callable = None, 
                    dialogue_format: str = "VisDial", dialogue_round: int = 0,
                    use_random_rounds: bool = False, use_caption_masking: bool = False, 
                    caption_masking_prob: float = 0.0, 
                    reference_image_dir: str = "./query_images",
                    **kwargs):

        super().__init__()
        self.json_file_path = Path(json_file_path)
        self.pil_transform = pil_transform
        self.dialogue_format = dialogue_format
        self.dialogue_round = dialogue_round
        self.use_random_rounds = use_random_rounds
        self.use_caption_masking = use_caption_masking
        self.caption_masking_prob = caption_masking_prob 

        self.reference_image_dir = Path(reference_image_dir)
        self.reference_filename_prefix = "train-" 

        self.reference_image_dir = Path(reference_image_dir)
        self.name_to_index_map = None

        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data: List[Dict] = json.load(f)
        

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
            item_info = self.data[index]
            target_path_str = item_info['img'] 
            dialog_list = item_info['dialog'] 
            
            max_allowed_round = self.dialogue_round

            if self.use_random_rounds:
                current_round_index = random.randint(0, max_allowed_round)

            else:
                current_round_index = max_allowed_round

            second_round_index = current_round_index # 默认相同

            caption = ""
            if self.dialogue_format == 'Summarized':
                if current_round_index >= len(dialog_list):
                    caption = dialog_list[-1]
                else:
                    caption = dialog_list[current_round_index]

            elif self.dialogue_format == 'VisDial':
                max_index = current_round_index + 1
                relevant_dialogs = dialog_list[:max_index]
                
                if self.use_caption_masking and \
                random.random() < self.caption_masking_prob and \
                len(relevant_dialogs) > 1:
                    relevant_dialogs = relevant_dialogs[1:] # 随机丢弃第一句
                
                caption = ", ".join(relevant_dialogs)


            target_filename_stem = Path(target_path_str).stem
            
            target_image = Image.open(target_path_str).convert("RGB")
            if self.pil_transform:
                target_image = self.pil_transform(target_image)
                
            ref_image_1 = self._load_ref_image(target_filename_stem, current_round_index)
            
            return ref_image_1, target_image, caption

    def _load_ref_image(self, target_filename_stem, round_idx):
            reference_filename = f"{self.reference_filename_prefix}{target_filename_stem}_{round_idx}.jpg"
            round_folder_name = f"round{round_idx}"
            reference_path = self.reference_image_dir / round_folder_name / reference_filename
            
            if not reference_path.exists():

                raise FileNotFoundError(f"Missing image: {reference_path}")
                
            image = Image.open(reference_path).convert("RGB")
            if self.pil_transform:
                image = self.pil_transform(image)
            return image
    
class CorpusDataset(Dataset):
    def __init__(self, json_file_path: str, pil_transform: callable = None):
        super().__init__()
        self.json_file_path = Path(json_file_path)
        self.pil_transform = pil_transform

        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.image_paths: List[str] = json.load(f)

        self.path_to_id_map: Dict[str, int] = {path: i for i, path in enumerate(self.image_paths)}
        print(f"CorpusDataset: 从 {json_file_path} 加载了 {len(self.image_paths)} 张图片。")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.pil_transform:
            image = self.pil_transform(image)
            
        return image_path, image
    
class ValidationQueriesDataset(Dataset):

    def __init__(self, queries_path: str, generated_image_dir: str):
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
        self.generated_image_dir = Path(generated_image_dir)
        self.dialog_length = 0 # 默认值，将由外部循环设置
        self.dialogue_format = Config.dialogue_format
        self.sep_token = ", "

    def __len__(self) -> int:
        return len(self.queries)

    def set_dialog_length(self, dialog_length: int):
        """由外部验证循环调用，用于设置当前要加载哪一轮的对话"""
        self.dialog_length = dialog_length

    def __getitem__(self, i: int) -> Dict:
        target_path = self.queries[i]['img']
        
        # text = self.queries[i]['dialog'][self.dialog_length]
        # text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        if self.dialogue_format == 'Summarized':
            text = self.queries[i]['dialog'][self.dialog_length]
        elif self.dialogue_format == 'VisDial':
            text = self.sep_token.join(self.queries[i]['dialog'][:self.dialog_length + 1])



        gen_image_filename = f"{i}_{self.dialog_length}.jpg"
        gen_image_path = (self.generated_image_dir / gen_image_filename).as_posix()

        return {
            'query_idx': i,       # 查询的原始索引
            'text': text,         # 当前轮次的文本
            'target_path': target_path, # 目标图片的真实路径
            'gen_path': gen_image_path    # 生成图片的路径
        }
    
class QueryImageDataset(Dataset):
    def __init__(self, queries: List[Dict], gen_image_dir: str, num_rounds: int, transform: callable = None):
        """
        用于批量加载生成的图像进行特征提取。
        """
        self.samples = []
        self.transform = transform
        gen_dir = Path(gen_image_dir)
        
        # 预先扫描所有需要的文件
        for query_idx in range(len(queries)):
            for round_idx in range(num_rounds):
                filename = f"{query_idx}_{round_idx}.jpg"
                filepath = gen_dir / filename
                
                # 检查文件是否存在
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
