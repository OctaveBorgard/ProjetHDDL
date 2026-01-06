import torch.optim as optim
import warnings
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import psutil
import json
import glob
import re
import datetime
import os


class OptimizationConfig:
    # Optimization setting
    optimizer = "AdamW"  # Options: adamw, adam, sgd, rmsprop
    initial_lr = 1e-3
    weight_decay = 1e-4
    gen_setting_kwargs = dict(lr=initial_lr, weight_decay=weight_decay)

    # adam and adamw specific parameters
    betas = (0.9, 0.999)
    eps = 1e-8

    # SGD specific parameters
    momentum = 0.9
    neserov = True

    # RMSprop specific parameters
    alpha = 0.99
    centered =False

    optim_kwargs = {"AdamW": dict(betas=betas, eps=eps),
                    "adam": dict(betas=betas, eps=eps),
                    "SGD": dict(momentum=momentum, neserov=neserov),
                    "RMSprop": dict(alpha=alpha, centered=centered, eps=eps, momentum=momentum)}

    # learning rate scheduler setting
    lr_scheduler = "CosineAnnealingLR"
    min_lr = 1e-6

    # CosineAnnealingLR specific parameters
    epochs = 60  #T_max for cosing scheduler

    # StepLR
    step_size = 30
    gamma = 0.1

    # ReduceLROnPlateau specific parameters
    factor = 0.1
    patience = 10
    mode = "min"

    # OneCycleLR specific parameters
    max_lr = 1e-2
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1e4
    
    scheduler_kwargs = {
        "CosineAnnealingLR": dict(eta_min=min_lr,T_max=epochs),
        "StepLR": dict(step_size=step_size, gamma=gamma),
        "ReduceLROnPlateau": dict(min_lr=min_lr, factor=factor, patience=patience, mode=mode),
        "OneCycleLR": dict(min_lr=min_lr, max_lr=max_lr, pct_start=pct_start,
                             div_factor=div_factor, final_div_factor= final_div_factor)
    }

    def get_optimizer(self, model, **kwargs):
        """
        Get the optimizer based on the configuration.
        Args:
            model: The model whose parameters need to be optimized
            **kwargs: Additional argument to the optimizer
        """
        params = model.parameters()
        if self.optim_kwargs[self.optimizer] is not None:
            optimizer_class = getattr(optim, self.optimizer)
            optimizer = optimizer_class(params, **self.gen_setting_kwargs, **self.optim_kwargs[self.optimizer], **kwargs)
            return optimizer
        else:
            warnings.warn(f"Optimizer {self.optimizer} not found, using Adamw instead")
            return optim.AdamW(params, lr=self.initial_lr, **kwargs)

    def get_scheduler(self, optimizer, **kwargs):
        if self.scheduler_kwargs[self.lr_scheduler] is not None:
            scheduler_class = getattr(optim.lr_scheduler, self.lr_scheduler)
            lr_scheduler = scheduler_class(optimizer, **self.scheduler_kwargs[self.lr_scheduler], **kwargs)
            return lr_scheduler
        else:
            warnings.warn(f"Learning rate scheduler {self.lr_scheduler} not found, Using constant lr instead")
            return optim.lr_scheduler.ConstantLR(optimizer, **kwargs)

import torch
class TrainingConfig:
    num_epochs: int=10
    batch_size: int=32
    device: str="cuda" if torch.cuda.is_available() else "cpu"

    def update(self, **kwargs):
        """
        Update the confiuration with new values
        Args:
        ***kwargs: New values for the configuration parameter
        """
        for key, val in kwargs.items():
            if not hasattr(self, key):
                warnings.warn(f"New argument {key}")
            setattr(self,key, val)

from collections import deque
class OnlineMovingAverage:
    def __init__(self, size=5000):
        self.size=size
        self.queue = deque(maxlen=size)
        self.sum = 0.0
        self.mean = 1.0
    def update(self, value):
        if len(self.queue) == self.size:
            self.sum -= self.queue[0]
        self.queue.append(value)
        self.sum += value
        self.mean = self.sum/len(self.queue)

def ema_avg_fn(averaged_model_parameter, model_parameter, n_averaged):
    decay = 0.99
    return decay * averaged_model_parameter + (1 - decay) * model_parameter

class LoggingConfig:
    # Logging frequencies
    log_freq = 10
    save_freq = 1

    # Tensorboard settings
    tensorboard = True
    tensorboard_dir = "runs" # Directory for tensorboard logs
    exp_name = "default" # Experiment name for the run

    # Checkpoint settings
    checkpoint_dir = "checkpoint"
    save_best_only = True
    save_last = True
    max_checkpoint = 5

    # Metrics to monitor
    monitor_metric = "val_loss"  # Metric to monitor
    monitor_mode = "min"        # 'min' for loss, 'max' for metrics like PSNR

    # Logging medias setting
    image_size = (256,256) # resize image in case the images do not have the same size
    image_logging = True
    log_image_freq = 200
    num_log_images= 4 
    log_loss_freq: int=10
    val_epoch_freq: int=5

    # System monitoring
    log_gpu_stats = False  # Log GPU utilization
    log_memory_stats = False    # Log memory usage

    def __init__(self, project_dir: str=None, exp_name: str="default", **kwargs ):
        self.writer = None
        self.project_dir = "" if project_dir is None else project_dir
        self.exp_dir = os.path.join(self.project_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.exp_dir, "runs")
        self.metadata_file = os.path.join(self.exp_dir,"metadata.json")

        metadata = None
        if os.path.exists(self.metadata_file):
            print(f"Loading metadata from {self.metadata_file}")
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

        if metadata is not None:
            self.global_step = metadata['global_step']
            self.epoch = metadata['epoch']
        
        else:
            self.global_step = 0
            self.epoch = 0
            

        for key, val in kwargs.items():
            if not hasattr(self, key):
                warnings.warn(f"Unknown argment {key}")
            setattr(self, key, val)
        

        self.best_metric = float("inf") if self.monitor_mode=="min" else -float("inf")

        if metadata is not None:
            if (self.monitor_metric == metadata['monitor_metric']) and (self.monitor_mode == metadata['monitor_mode']):
                self.best_metric = metadata['best_metric']


    def initialize(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        else:
            self.writer = None
        
        self._save_metadata()

    def _save_metadata(self):
        """Save metadata about the training process"""

        metadata = {
            "exp_name": self.exp_name,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "tensorboard_logdir": self.tensorboard_dir if self.tensorboard else None
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    def load_metadata(self):
        """Load metadata about the training process"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            return metadata
        else:
            warnings.warn("No metadata file found, starting from scratch, returning None")
            return None
    
    def get_checkpoint_path(self, epoch, metric_value=None):
        """
        Return checkpoint path based on epoch and metric value
        """
        if metric_value is not None:
            return os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}_{self.monitor_metric}_{metric_value:.4f}.pth")
        else:
            return os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")
        
    def save_checkpoint(self, state, epoch, metric_value=None):
        """
        Save a checkpoint with training state.

        Args:
            state (dict): State dictionary containing model, optimizwer state etc.
            epoch (int): Current epoch
            metric_value (float): Current value of the monitored metric
        """

        # Update best metric if applicable
        if metric_value is not None:
            is_best = (self.monitor_mode=='min' and metric_value < self.best_metric
                       ) or (self.monitor_mode=='max' and metric_value > self.best_metric)
            if is_best:
                self.best_metric = metric_value
        
        checkpoint_path = self.get_checkpoint_path(epoch, metric_value)
        state.update({
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric
        })

        torch.save(state, checkpoint_path)
        self._save_metadata()
        
        return checkpoint_path
    
    def load_best_checkpoint(self, chekpoint_path: str=None, metric:str=None, mode:str=None, verbose: bool=True):
        """
        Load a checkpoint file (optionally selecting the best one by a monitored metric)
        according to the requested mode ("min" or "max").

        Parameters
        ----------
        chekpoint_path : str or None
            Path to a specific checkpoint file to load.
        metric : str or None, optional
            Name of the metric used to find matching checkpoint files.
        mode : {"min", "max"} or None, optional
        verbose : bool, optional
        
        Returns
        -------
        dict or None
        """
        metric = self.monitor_metric if metric is None else metric
        mode = self.monitor_mode if ((mode is None) or (mode not in ["min","max"])) else mode
        if chekpoint_path is None:
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, f"*{metric}_*.pth"))
            if not checkpoints:
                warnings.warn("No checkpoint found in the experiment directory, load the latest checkpoint instead")

                return self.load_latest_checkpoint(verbose=verbose)
            else:
                def get_metric_value(checkpoint_path):
                    match = re.search(rf"{metric}_([0-9]+(?:\.[0-9]+)?)", checkpoint_path)
                    if match:
                        return float(match.group(1))
                    else:
                        return float("inf") if mode=="min" else float("-inf")
                if mode == "min":
                    checkpoint_path = min(checkpoints, key=get_metric_value)
                else:
                    checkpoint_path = max(checkpoints, key=get_metric_value)
        else:
            if not os.path.exists(checkpoint_path):
                warnings.warn(f"Checkpoint {checkpoint_path} does not exist.")
                return None
            
        state = torch.load(checkpoint_path)
        self.global_step = state.get("global_step",0) + 1 
        self.epoch = state.get("epoch", 0) + 1
        self.best_metric = state.get("best_metric", self.best_metric)

        if verbose:
            print(f"Loaded checkpoint from: {checkpoint_path}")
        return state

    
    def load_latest_checkpoint(self, checkpoint_path=None, verbose=True):
        """Load a checkpoint and return the training state. if no checkpoint_path is provide,
        load the most recent checkpoint. If no checkpoint exists, return None.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file. If None, load the latest checkpoint
            verbose (bool): whetther to print the checkpoint_path 
        
        Returns:
            dict: The loaded state dictionary, or None if no checkpoint exists
        """
        if checkpoint_path is None:
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "epoch_*.pth"))
            
            if not checkpoints:
                warnings.warn("No checkpoint found in the experiment directory.")
                return None
            else:
                def get_epoch_num(checkpoint_path):
                    match = re.search(r"epoch_(\d+)", checkpoint_path)
                    return int(match.group(1)) if match else -1
                
                checkpoint_path = max(checkpoints, key=get_epoch_num)

        else:
            if not os.path.exists(checkpoint_path):
                warnings.warn(f"Checkpoint {checkpoint_path} does not exist.")
                return None
            
        state = torch.load(checkpoint_path)
        self.global_step = state.get("global_step",0) + 1 
        self.epoch = state.get("epoch", 0) + 1
        self.best_metric = state.get("best_metric", self.best_metric)

        if verbose:
            print(f"Loaded checkpoint from: {checkpoint_path}")
            print(f"Resuming from epoch {self.epoch}")
        
        return state
    
    def log_metrics(self, metrics, step="None", prefix="train"):
        """
        Log metrics to TensorBoard
        Args:
            metrics (dict): Dictionary of metric names and values
            step (int, optional): Current step within the epoch
            prefix (str): Prefix for metrix name (e.g., `train` or `val`)
        """
        if self.writer is None:
            return
        else:
            step = self.global_step if step is None else step

            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{name}", value, step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.writer.add_scalar(f"{prefix}/{name}", value.item(), step)
    
    def log_histogram(self, values, name, step=None):
        """Log a histogram of values to TensorBoard
        Args:
            values (torch.Tensor): Tensor of values to log
            name (str): Name for the histogram
            step (int, optional): Current step within the epoch
        """
        if self.writer is None:
            return
        else:
            step = self.global_step if step is None else step
            self.writer.add_histogram(name, values, step)
    
    def clean_old_checkpoint(self):
        """
        Keep only the top checkpoints based on the monitored metric
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir,"epoch*.pth"))

        if len(checkpoints) >= self.max_checkpoint:
            checkpoints_metrics = []
            metric_pattern = re.compile(rf"{self.monitor_metric}_([0-9.]+)(?=\.pth)")
            for checkpoint in checkpoints:
                match = metric_pattern.search(checkpoint)
                if match:
                    metric_value = float(match.group(1))
                    checkpoints_metrics.append((checkpoint, metric_value))
                else:
                    # Use creation time as fallback
                    creation_time = os.path.getctime(checkpoint)
                    checkpoints_metrics.append((checkpoint, creation_time))
                
            reverse = True if self.monitor_mode == 'max' else False 
            checkpoints_metrics.sort(key=lambda x: x[1], reverse=reverse)
            for checkpoint, _ in checkpoints_metrics[self.max_checkpoint:]:
                os.remove(checkpoint)
    
    def log_images(self, images_dict, step):
        """
        Log images to Tensorboard
        Args:
            images_dict (dictionary): Dictionary of image names and tensors,
            step (int, optional): Current step 
        """

        if (self.writer is None) or (not self.image_logging):
            return
        else:
            step = self.global_step if step is None else step
            for name, images in images_dict.items():
                if isinstance(images, torch.Tensor):
                    if images.ndim  == 3:
                        images = images.unsqueeze(0)
                    
                    images = images[:self.num_log_images]
                    self.writer.add_images(name, images, step)
                self.writer.flush()
    
    def log_figure(self, figure, name, step):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure object
            name (str): Name for the figure
            step (int): Current step/iteration
        """
        if self.writer is None:
            return
        step = self.global_step if step is None else step
        self.writer.add_figure(name, figure, global_step=step)   

    
    def log_system_stats(self, step):
        """
        Log system statistics to TensorBoard.

        Args:
            writer: TensorBoard writer instance
            step (int): Current step/iteration
        """
        if self.writer is None:
            return

        step = self.global_step if step is None else step
        if self.log_gpu_stats and torch.cuda.is_available():
            self.writer.add_scalar(
                "system/gpu_utilization", torch.cuda.utilization(), step
            )
            self.writer.add_scalar(
                "system/gpu_memory_allocated", torch.cuda.memory_allocated(), step
            )

        if self.log_memory_stats:
            self.writer.add_scalar(
                "system/ram_usage_percent", psutil.virtual_memory().percent, step
            )

        self.writer.flush()

    def log_hyperparameters(self, hparams, main_key: str = "hyperparameters"):
        """
        Log hyperparameters to TensorBoard.

        Args:
            hparams (dict): Dictionary of hyperparameters
        """
        if self.writer is None:
            return

        # Log hyperparameters as text
        for key, value in hparams.items():
            self.writer.add_text(f"{main_key}/{key}", str(value))

        self.writer.flush()
            



def move_to_device(images: list=None, labels: list=None, device="cpu", segmentation: bool=False):
    if images is not None:
        images = [img.to(device) for img in images]
    if labels is not None:
        if segmentation:
            labels = [label.to(device) for label in labels]
        else:
            labels = [torch.tensor(label, device=device) for label in labels]
    return images, labels    

def collate_fn(batch):
    return tuple(zip(*batch)) 



### loss functions ###

class DiceLoss(torch.nn.Module):
    def __init__(self,
                 smooth: float=1e-6,
                 reduction: str="mean",
                 ignore_index: int=None,
                 need_softmax: bool=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.need_softmax = need_softmax

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = preds.shape[1]
        
        if self.need_softmax:
            inputs = F.softmax(preds, dim=1)
        else:
            inputs = preds
        
        # targets: (B, 1, H, W) -> (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets.squeeze(1).long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        if self.ignore_index is not None:
            selected_indices = [i for i in range(num_classes) if i != self.ignore_index]
            inputs = inputs[:,selected_indices,:,:]
            targets_one_hot = targets_one_hot[:,selected_indices,:,:]
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3)) #(B, C)
        
        cardinality = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) #(B, C)
        
        # Calculate Dice per class
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_score = dice_score.mean(dim=-1)

        if self.reduction == "mean":
            dice_score = dice_score.mean()
        elif self.reduction == "sum":
            dice_score = dice_score.sum()
        
        return 1 - dice_score
        
class DiceCELoss(torch.nn.Module):
    def __init__(self,
                 ce_weight: float=0.5,  
                 dice_weight: float=0.5,
                 need_softmax: bool=True,
                 reduction: str="mean",
                 ignore_index: int=None,
                 smooth: float=1e-6):
        super(DiceCELoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(reduction=reduction, ignore_index=ignore_index, smooth=smooth, need_softmax=need_softmax)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.need_softmax = need_softmax

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.need_softmax:
            ce_loss = self.ce_loss(F.softmax(preds, dim=1), targets.squeeze(1).long())
            dice_loss = self.dice_loss(F.softmax(preds, dim=1), targets)
        else:
            ce_loss = self.ce_loss(preds, targets.squeeze(1).long())
            dice_loss = self.dice_loss(preds, targets)

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss


        




