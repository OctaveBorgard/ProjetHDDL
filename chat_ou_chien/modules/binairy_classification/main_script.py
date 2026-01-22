#%%
import sys, os
project_abs_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..","..")
)
sys.path.append(project_abs_dir)
from utils import (create_df,
                   CatDogBinary,
                   CatDogBreed,
                   CatDogSegmentation,
                   show_images)
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from deepinv.utils import plot

# %%
from torch.utils.data import random_split, RandomSampler, DataLoader
from training import (collate_fn,
                      LoggingConfig,
                      TrainingConfig)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_test_ratio = 0.9
batch_size = 64
save_dir = "exp/binaryclassification"

#################### DEFINE DATASET ##############################
df_train = create_df(cls_list_path=os.path.join(project_abs_dir, "data/annotations/trainval.txt"),
                image_path=os.path.join(project_abs_dir, "data/images"))

df_test = create_df(cls_list_path=os.path.join(project_abs_dir, "data/annotations/test.txt"),
                image_path=os.path.join(project_abs_dir, "data/images"))

train_dataset = CatDogBinary(df=df_train)
test_dataset = CatDogBinary(df=df_test)
class_str = CatDogBinary.species

train_size = len(train_dataset)
test_size = len(test_dataset)
print(f"Train set size: {train_size}, test set size: {test_size}")



transform_eval =  v2.Compose([
    v2.Resize((224,224)),
    v2.ToDtype(dtype=torch.float32, scale=True)
])

train_dataset.transform = transform_eval
test_dataset.transform = transform_eval 
if train_size < batch_size:
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=batch_size)
    shuffle = False
else:
    sampler = None
    shuffle = True

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                shuffle=shuffle, pin_memory=True, sampler=sampler, drop_last=False,
                                num_workers=8)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)

########################### DEFINE MODEL ########################
model_kwargs = dict(weights=EfficientNet_B0_Weights.DEFAULT)
model = efficientnet_b0(**model_kwargs)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_str))



######################### training logger ##################

logger = LoggingConfig(project_dir=os.path.join(project_abs_dir, save_dir),
                        exp_name=f"EfficientNet_{train_size}")
logger.monitor_metric = "train_avg_loss"
logger.monitor_mode = "min"

# state = logger.load_checkpoint()
state = torch.load(os.path.join(project_abs_dir, "exp/binaryclassification/EfficientNet/checkpoints/epoch_504_test_avg_loss_0.00060.pth"))
model.load_state_dict(state['model_state_dict'])
model.to(device)
model.eval()

train_accuracy = 0
test_accuracy = 0

for images, labels in train_loader:
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels, device=device)

    with torch.no_grad():
        pred_labels = model(images).detach().argmax(dim=1, keepdim=False)
    indice_fail = labels != pred_labels
    if torch.sum(indice_fail) > 0:
        show_images(images[indice_fail], title=[f"T:{labels[i].item()}, P:{pred_labels[i].item()}" for i in torch.where(indice_fail)[0]])
    train_accuracy += torch.sum(labels == pred_labels).item()
train_accuracy = train_accuracy/train_size

for images, labels in val_loader:
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels, device=device)

    with torch.no_grad():
        pred_labels = model(images).detach().argmax(dim=1, keepdim=False)
    indice_fail = labels != pred_labels
    if torch.sum(indice_fail) > 0:
        show_images(images[indice_fail], title=[f"T:{labels[i].item()}, P:{pred_labels[i].item()}" for i in torch.where(indice_fail)[0]])

    test_accuracy += torch.sum(labels == pred_labels).item()
test_accuracy = test_accuracy/test_size

print(f"Accuracy obtained on train set: {train_accuracy:.3f}, and on test set: {test_accuracy:.3f}")




# %%
