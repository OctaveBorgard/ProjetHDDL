#%%
import os, sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_path)
from utils import (CatDogSegmentation,
                    create_df)
from models import Unet_Segmenter
import torch.utils.data as data
from training import collate_fn, DiceCELoss
from torchvision.transforms import v2
import numpy as np
import torch
# %%
train_test_ratio = 0.9
data_path = os.path.join(project_path, "data")
df = create_df(cls_list_path=os.path.join(data_path, "annotations/list.txt"),
                image_path=os.path.join(data_path, "images"),
                segmentation_annot_path=os.path.join(data_path, "annotations/trimaps"))

ramdom_state = np.random.RandomState(seed=42)
df_train = df.sample(frac=train_test_ratio, random_state=ramdom_state)
df_test = df.drop(df_train.index).reset_index(drop=True)
df_train = df_train.reset_index(drop=True)

dataset_train = CatDogSegmentation(df_train)
dataset_test = CatDogSegmentation(df_test)
class_str = ["background", "cat", "dog"]

train_size = len(dataset_train)
test_size = len(dataset_test)
print(f"Train set size: {train_size}, test set size: {test_size}")

transform_train = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomGrayscale(p=0.1),
    v2.GaussianNoise(),
    v2.ColorJitter(),
    v2.RandomCrop((224,224), pad_if_needed=False),
    v2.ToDtype(dtype=torch.float32, scale=True)
])

transform_test =  v2.Compose([
    v2.RandomCrop((224,224), pad_if_needed=True),
    v2.ToDtype(dtype=torch.float32, scale=True)
])
dataset_train.transform = transform_test
dataset_test.transform = transform_test

# %%
model_kwargs = dict(layers_per_block=2,
                    non_linearity="silu",
                    skip_connection=True,
                    center_input_sample=True)
model = Unet_Segmenter(**model_kwargs)

print(f"Number of trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f} Million")

# %%
num_classes = 3
from torchmetrics.segmentation import MeanIoU, DiceScore

val_loader = data.DataLoader(dataset_test, batch_size=5, shuffle=True, collate_fn=collate_fn, drop_last=False)


mean_iou_metric_test = MeanIoU(num_classes=num_classes, input_format='index')

images_test, labels_test = next(iter(val_loader))
images_test = torch.stack(images_test) 
labels_test = torch.stack(labels_test)

pred_labels_test = model(images_test)
mean_iou_test = mean_iou_metric_test(pred_labels_test.argmax(dim=1, keepdim=True).long(), labels_test.long())


# %%
criterion = DiceCELoss(ce_weight=0.5,
                        dice_weight=0.5,
                        need_softmax=True,
                        reduction="mean",
                        ignore_index=None,
                        smooth=1e-6)

loss = criterion(pred_labels_test, labels_test)
# %%
loss.item()

# %%
