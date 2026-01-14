#%%
import torch
import torch.nn as nn

from torchmetrics.segmentation import MeanIoU, DiceScore
from torchvision.transforms import v2
from tqdm import tqdm
from train_utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, ema_avg_fn
from torch.optim.swa_utils import AveragedModel
import os
import sys
import numpy as np
import deepinv as dinv
import matplotlib
from typing import Callable
matplotlib.use("Agg")


# %%
def training_loop(
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    logger: LoggingConfig,
    class_str: list[str]
):
    # Initialize EMA for model weights using PyTorch's AveragedModel
    swa_start = 1000 # Start using SWA after 1000 iterations

    model = model.to(config.device)
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
    ema_model = ema_model.to(config.device)
    state = logger.load_latest_checkpoint()
    if state is not None:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        ema_model.load_state_dict(state['ema_model_state_dict'])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
    else:
        global_step = 0
        start_epoch = 0
    
    num_classes = len(class_str)

    logger.global_step = global_step


    train_avg_loss = OnlineMovingAverage(size=5000)
    
    test_avg_loss = OnlineMovingAverage(size=1000)


    mean_iou_metric_train = MeanIoU(num_classes=num_classes, input_format='index').to(config.device)
    dice_score_metric_train = DiceScore(num_classes=num_classes, input_format='index').to(config.device)

    mean_iou_metric_test = MeanIoU(num_classes=num_classes, input_format='index').to(config.device)
    dice_score_metric_test = DiceScore(num_classes=num_classes, input_format='index').to(config.device)


    for epoch in range(start_epoch, config.num_epochs):
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10)
        for images, labels in pb:
            # images has to be is a list of tensor with the same size
            # targets is a list of tensor
            model.train()
            images = torch.stack(images).to(config.device)
            labels = torch.stack(labels).to(config.device).long()


            pred_labels = model(images)
            loss = criterion(pred_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            if logger.global_step > swa_start and logger.global_step % 5 == 0:
                ema_model.update_parameters(model)
            
            train_avg_loss.update(loss.item()/len(images))
            pb.set_description(f"Avg_training_loss: {train_avg_loss.mean:.3e}")
            
            if ((logger.global_step + 1) % logger.log_loss_freq == 0) or (logger.global_step == 0):
                
                with torch.no_grad():
                    images_test, labels_test =next(iter(val_loader))
                    images_test = torch.stack(images_test).to(config.device) 
                    labels_test = torch.stack(labels_test).to(config.device)

                    pred_labels_test = model(images_test)
                    test_loss = criterion(pred_labels_test, labels_test)
                    mean_iou_metric_test.update(pred_labels_test.argmax(dim=1, keepdim=True).long(), labels_test.long())
                    dice_score_metric_test.update(pred_labels_test.argmax(dim=1, keepdim=True).long(), labels_test.long())


                    mean_iou_metric_train.update(pred_labels.argmax(dim=1, keepdim=True).long(), labels.long())
                    dice_score_metric_train.update(pred_labels.argmax(dim=1, keepdim=True).long(), labels.long())
                
                test_avg_loss.update(test_loss.item())

                metrics = {
                    "val_loss": test_avg_loss.mean,
                    "val_mean_iou": mean_iou_metric_test.compute().item(),
                    "val_dice_score": dice_score_metric_test.compute().item(),
                    "train_loss": train_avg_loss.mean,
                    "train_mean_iou": mean_iou_metric_train.compute().item(),
                    "train_dice_score": dice_score_metric_train.compute().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": grad_norm.max()
                }

                logger.log_metrics(metrics, logger.global_step)
                logger.log_histogram(grad_norm, "grad_norm", logger.global_step)
                

            if ((logger.global_step+1) % logger.log_image_freq == 0) or (logger.global_step == 0):
                model.eval()
                num_log_images = logger.num_log_images
                train_images = images[:num_log_images]
                train_labels = labels[:num_log_images]
                train_pred_labels = pred_labels[:num_log_images].detach()

                test_images, test_labels = next(iter(val_loader))
                test_images = torch.stack(test_images[:num_log_images]).to(config.device)
                test_labels = torch.stack(test_labels[:num_log_images]).to(config.device)
                with torch.no_grad():
                    test_pred_labels = model(test_images).detach()
                                
                fig = dinv.utils.plot([train_images.cpu(), 
                                      train_labels.cpu(), 
                                      train_pred_labels.argmax(dim=1, keepdim=True).cpu(),
                                      test_images.cpu(),
                                      test_labels.cpu(),
                                      test_pred_labels.argmax(dim=1, keepdim=True).cpu()],
                                     titles=["train Image", "train gt mask", "train pred mask",
                                             "test Image", "test gt mask", "test pred mask"],
                                     return_fig=True,   
                                     show=False)
                logger.log_figure(figure=fig,name="samples from train and test sets",step=logger.global_step)
                


            logger.global_step += 1

        # Save checkpoint
        if ((epoch + 1) == config.num_epochs) or (epoch % logger.save_freq == 0):
            state = {
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "global_step": logger.global_step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }

            logger.save_checkpoint(state, epoch, metric_value=dice_score_metric_test.compute().item())
            if  epoch % logger.save_freq == 0:
                logger.clean_old_checkpoint()

                

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils import (CatDogSegmentation,
                       create_df)
    from models import Unet_Segmenter
    import torch.utils.data as data
    import argparse  
    from train_utils import collate_fn, DiceCELoss

    parser = argparse.ArgumentParser(description="Training script for Unet_Segmenter task.")
    parser.add_argument("--train_test_ratio", type=float, default=0.8, help="The train test split ratio")
    parser.add_argument("--num_epochs", type=int, default=300, help="Numeber of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, help="Saved directory",
                        default='/home/bao/School/5A/HDDL/bacteria_detection_app/exp/default')
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for training and testing")
    args = parser.parse_args()

    #################### DEFINE DATASET ##############################
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_path, "data")

    df = create_df(cls_list_path=os.path.join(data_path, "annotations/list.txt"),
                   image_path=os.path.join(data_path, "images"),
                   segmentation_annot_path=os.path.join(data_path, "annotations/trimaps"))
    
    ramdom_state = np.random.RandomState(seed=42)
    df_train = df.sample(frac=args.train_test_ratio, random_state=ramdom_state)
    df_test = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    
    dataset_train = CatDogSegmentation(df_train)
    dataset_test = CatDogSegmentation(df_test)
    class_str = ["background", "cat", "dog"]

    train_size = len(dataset_train)
    test_size = len(dataset_test)
    print(f"Train set size: {train_size}, test set size: {test_size}")

    transform_train = v2.Compose([
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        # v2.RandomGrayscale(p=0.1),
        # v2.GaussianNoise(),
        v2.ColorJitter(),
        v2.RandomCrop((args.crop_size,args.crop_size), pad_if_needed=True, fill=1),
    ])

    transform_test =  v2.Compose([
        v2.RandomCrop((args.crop_size,args.crop_size), pad_if_needed=True, fill=1),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])
    dataset_train.transform = transform_train
    dataset_test.transform = transform_test

    if train_size < args.batch_size:
        sampler = data.RandomSampler(dataset_train, replacement=True, num_samples=args.batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=collate_fn,
                                   shuffle=shuffle, pin_memory=True, sampler=sampler, drop_last=False,
                                   num_workers=8)
    val_loader = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)


    ########################### DEFINE MODEL ############################
    model_kwargs = dict(layers_per_block=2,
                        block_out_channels=(16, 64, 128),
                        non_linearity="silu",
                        skip_connection=True,
                        center_input_sample=True)
    model = Unet_Segmenter(**model_kwargs)

    print(f"Number of trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f} Million")

    ######################### Training Config and training logger ##################


    # Create training configuration
    training_config = TrainingConfig()
    # training_config.device = "cpu"
    training_config.update(**vars(args))

    # Create logging configuration
    monitor_metric = "val_dice_score"
    monitor_mode = "max"
    num_log_images = 3

    # Save checkpoint every 400 steps
    num_step_per_epoch = max(len(train_loader), 1)
    freq = max(1, int(400 // num_step_per_epoch))
    save_freq = freq
    val_epoch_freq = freq
    log_loss_freq = 5
    log_image_freq = 200

    logger_kwargs = dict(monitor_metric=monitor_metric,
                         monitor_mode=monitor_mode,
                         save_freq=save_freq,
                         val_epoch_freq=val_epoch_freq,
                         log_loss_freq=log_loss_freq,
                         log_image_freq=log_image_freq,
                         num_log_images=num_log_images)
    
    logger = LoggingConfig(project_dir=os.path.join(project_path, args.save_dir),
                        exp_name=f"Unet_Segmenter_crop_{args.crop_size}",
                        **logger_kwargs)
    logger.initialize()
    logger.log_hyperparameters(vars(args), main_key="training_config")

    optim_config = OptimizationConfig()
    optimizer = optim_config.get_optimizer(model)
    lr_scheduler = optim_config.get_scheduler(optimizer)

    criterion = DiceCELoss(ce_weight=0.5,
                            dice_weight=0.5,
                            need_softmax=True,
                            reduction="mean",
                            ignore_index=None,
                            smooth=1e-6)

    ########################## Lance training loop ###############################
    training_loop(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  config=training_config,
                  logger=logger,
                  class_str=class_str)





