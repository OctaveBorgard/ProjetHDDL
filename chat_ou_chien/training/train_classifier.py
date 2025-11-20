#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from tqdm import tqdm
from train_utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, ema_avg_fn, move_to_device
from torch.optim.swa_utils import AveragedModel
import os
import sys
import numpy as np
import deepinv as dinv

# %%
def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    preprocess: v2,
    lr_scheduler: torch.optim.lr_scheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    logger: LoggingConfig,
    class_str: list[str]=None
):
    # Initialize EMA for model weights using PyTorch's AveragedModel
    swa_start = 1000 # Start using SWA after 1000 iterations

    model = model.to(config.device)
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
    ema_model = ema_model.to(config.device)
    state = logger.load_checkpoint()
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
    
    logger.global_step = global_step
    train_avg_loss = OnlineMovingAverage(size=5000)
    test_avg_loss = OnlineMovingAverage(size=1000)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, config.num_epochs):
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10)
        for images, labels in pb:
            # images has to be is a list of tensor with the same size
            # targets is a list of tensor
            model.train()
            images = torch.stack(images).to(config.device)
            images_transformed = preprocess(images)
            labels = torch.tensor(labels, device=config.device)

            pred_labels = model(images_transformed)
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
                    images_test_transformed = preprocess(images_test)
                    labels_test = torch.tensor(labels_test, device=config.device)

                    pred_labels_test = model(images_test_transformed)
                    test_loss = criterion(pred_labels_test, labels_test).item()
                
                test_avg_loss.update(test_loss/len(images_test))
                del images_test,  labels_test, pred_labels_test, test_loss
                torch.cuda.empty_cache()    

                metrics = {
                    "val_loss": test_avg_loss.mean,
                    "train_loss": train_avg_loss.mean,
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": grad_norm.max()
                }

                logger.log_metrics(metrics, logger.global_step)
                logger.log_histogram(grad_norm, "grad_norm", logger.global_step)

            if ((logger.global_step+1) % logger.log_image_freq == 0) or (logger.global_step == 0):
                model.eval()
                num_log_images = logger.num_log_images
                train_samples = images[:num_log_images]
                true_labels = labels[:num_log_images].tolist()
                pred_labels = pred_labels[:num_log_images].argmax(dim=1, keepdim=False).tolist()

                if class_str is not None:
                    titles = [f"{class_str[int(i)]} : {class_str[int(j)]}" for i,j in zip(true_labels, pred_labels)]
                else:
                    titles = [f"{int(i)} : {int(j)}" for i,j in zip(true_labels, pred_labels)]
                
                fig = dinv.utils.plot(img_list=list(train_samples.unbind(0)),
                                            titles=titles,
                                            return_fig=True,
                                            show=False)
                logger.log_figure(figure=fig,name="sample from train set",step=logger.global_step)
                
                # Log images of validation set
                images_test, labels_test = next(iter(val_loader))
                images_test = torch.stack(images_test[:num_log_images]).to(config.device)
                images_val_transformed = preprocess(images_test)

                labels_test = labels_test[:num_log_images]
                pred_labels_val = model(images_val_transformed).argmax(dim=1, keepdim=False).detach().tolist()

                if class_str is not None:
                    titles = [f"{class_str[int(i.item())]} : {class_str[int(j)]}" for i,j in zip(labels_test, pred_labels_val)]
                else:
                    titles = [f"{int(i.item())} : {int(j)}" for i,j in zip(labels_test, pred_labels_val)]
                
                fig = dinv.utils.plot(img_list=list(images_test.unbind(0)),
                                            titles=titles,
                                            return_fig=True,
                                            show=False)
                logger.log_figure(figure=fig,name="sample from test set",step=logger.global_step)
                


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

            logger.save_checkpoint(state, epoch, metric_value=test_avg_loss.mean)
            if  epoch % logger.save_freq == 0:
                logger.clean_old_checkpoint()

                

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils import (CatDogBinary,
                       CatDogBreed,
                       create_df)
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    import torch.utils.data as data
    from torch.utils.data import random_split
    import argparse  
    from train_utils import collate_fn

    parser = argparse.ArgumentParser(description="Training script for EfficienttNetB0 for Dog and Cat classification task.")
    parser.add_argument("--problem_type", type=str, default="binaryclassification",
                        choices=['binaryclassification', 'finclassification'])
    parser.add_argument("--train_test_ratio", type=float, default=0.8, help="The train test split ratio")
    parser.add_argument("--num_epochs", type=int, default=300, help="Numeber of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, help="Saved directory",
                        default='/home/bao/School/5A/HDDL/bacteria_detection_app/exp/default')
    args = parser.parse_args()

    #################### DEFINE DATASET ##############################
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_path, "data")

    df = create_df(cls_list_path=os.path.join(data_path, "annotations/list.txt"),
                   image_path=os.path.join(data_path, "images"))
    if args.problem_type.lower() == "binaryclassification":
        dataset_full = CatDogBinary(df, transforms=v2.Resize((224,224)))
        class_str = dataset_full.species
    elif args.problem_type.lower() == "fineclassification":
        dataset_full = CatDogBreed(df, transforms=v2.Resize((224,224)))
        class_str = dataset_full.breeds
    else:
        raise ValueError("the problem type must be \"binary classifiction\" or \"fineclassification\"")
    
    

    train_size = int(args.train_test_ratio*len(dataset_full))
    test_size = len(dataset_full) - train_size
    print(f"Train set size: {train_size}, test set size: {test_size}")

    g = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset_full,
                                               [train_size, test_size],
                                               generator=g)

    if train_size < args.batch_size:
        sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=args.batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                   shuffle=shuffle, pin_memory=True, sampler=sampler, drop_last=False,
                                   num_workers=8)
    val_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)


    ########################### DEFINE MODEL ########################
    model_kwargs = dict(weights=EfficientNet_B0_Weights.DEFAULT)
    model = efficientnet_b0(**model_kwargs)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_str))
    # Train only the classification head
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    print("Number of trainable parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

    preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()


    ######################### Training Config and training logger ##################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create training configuration
    training_config = TrainingConfig()
    training_config.update(**vars(args))

    # Create logging configuration
    logger = LoggingConfig(project_dir=os.path.join(project_path, args.save_dir),
                           exp_name=f"EfficientNet_{train_size}")
    logger.monitor_metric = "train_avg_loss"
    logger.monitor_mode = "min"
    logger.num_log_images = 5
    logger.initialize()
    logger.log_hyperparameters(vars(args), main_key="training_config")


    # Save checkpoint every 400 steps
    num_step_per_epoch = max(len(train_loader), 1)
    freq = max(1, int(400 // num_step_per_epoch))
    logger.save_freq = freq
    logger.val_epoch_freq = freq
    logger.log_loss_freq = 5
    logger.log_image_freq = 200

    optim_config = OptimizationConfig()
    optimizer = optim_config.get_optimizer(model)
    lr_scheduler = optim_config.get_scheduler(optimizer)


    ########################## Lance training loop ###############################
    training_loop(model=model,
                  optimizer=optimizer,
                  preprocess=preprocess,
                  lr_scheduler=lr_scheduler,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  config=training_config,
                  logger=logger,
                  class_str=class_str)




# %%
