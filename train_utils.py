from pathlib import Path

import torch
from torch import nn, optim
from torchvision import transforms

from timm.utils import ModelEma
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation


def build_beit3_transform(is_train: bool, config: dict):
    """
    Build torchvision transforms for BEiT-3 pre-processing.
    """
    if is_train:
        # Train mode: with data augmentation
        t = [
            RandomResizedCropAndInterpolation(config['input_size'], scale=(0.5, 1.0), interpolation=config['train_interpolation']), 
            transforms.RandomHorizontalFlip(),
        ]
        if config.get('randaug', False): 
            t.append(transforms.RandAugment(num_ops=2, magnitude=9))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        return transforms.Compose(t)
    else:
        # Validation mode: deterministic resizing
        return transforms.Compose([
            transforms.Resize((config['input_size'], config['input_size']), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

def save_checkpoint(name: str, cur_epoch: int, model: nn.Module, optimizer: optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, best_metric: float, training_path: Path,
                    scheduler, model_ema: ModelEma = None):
    """
    Save model weights, optimizer, and scaler states to a .pt file.
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        'epoch': cur_epoch,
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_recall_at_10': float(best_metric)
    }
    if model_ema is not None:
        checkpoint['model_ema_state_dict'] = model_ema.ema.state_dict()
        
    torch.save(checkpoint, str(models_path / f'{name}.pt'))

def update_train_running_results(train_running_results: dict, loss: torch.Tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu', non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch

def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    # Prevent division by zero with max(1, ...)
    avg_loss = train_running_results['accumulated_train_loss'] / max(1, train_running_results['images_in_epoch'])
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] train loss: {avg_loss:.3f} "
    )