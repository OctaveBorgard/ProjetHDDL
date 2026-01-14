#%%
import os, sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_path)
from utils import (CatDogSegmentation,
                    create_df,
                    show_images)
from models import Unet_Segmenter
import torch.utils.data as data
from training import collate_fn, DiceCELoss
from torchvision.transforms import v2
from torchvision import transforms as T
import numpy as np
import torch
from training import LoggingConfig
from torchmetrics.segmentation import MeanIoU, DiceScore
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
class_str = ["foreground", "background", "not classified"]

train_size = len(dataset_train)
test_size = len(dataset_test)
print(f"Train set size: {train_size}, test set size: {test_size}")


transform =  v2.Compose([
    # v2.RandomCrop((224,224), pad_if_needed=True),
    v2.ToDtype(dtype=torch.float32, scale=True)
])
dataset_train.transform = transform
dataset_test.transform = transform

train_loader = data.DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = data.DataLoader(dataset_test, batch_size=16, shuffle=False, collate_fn=collate_fn)

# %%
model_kwargs = dict(layers_per_block=2,
                    block_out_channels=(16, 64, 128),
                    non_linearity="silu",
                    skip_connection=True,
                    center_input_sample=True)
model = Unet_Segmenter(**model_kwargs)


save_dir = "exp/segmenter"
crop_size = 224
monitor_metric = "val_dice_score"
monitor_mode = "max"

logger = LoggingConfig(project_dir=os.path.join(project_path, save_dir),
                    exp_name=f"Unet_Segmenter_crop_{crop_size}",
                    monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode)
state = logger.load_best_checkpoint()

model.load_state_dict(state['model_state_dict'])
model.to(device)
model.eval()
# %% test inference on a sample



mean_iou_metric_test = MeanIoU(num_classes=len(class_str), input_format='index').to(device)
mean_dice_metric_test = DiceScore(num_classes=len(class_str), average='macro', input_format='index').to(device)
for i in range(len(dataset_test)):
    img, mask = dataset_test[i]
    img = img.to(device)
    mask = mask.to(device)
    input_image = img.unsqueeze(0) 
    with torch.no_grad():
        output_mask = model.inference(input_image, sliding_window_size=224, device=device)
    mean_iou_metric_test.update(output_mask.long(), mask.unsqueeze(0).long())
    mean_dice_metric_test.update(output_mask.long(), mask.unsqueeze(0).long())
    
    # show_images([img, mask, output_mask.squeeze(0)], title=['Input Image', 'Ground Truth Mask', 'Predicted Mask'])

print(f"Mean IoU on test set: {mean_iou_metric_test.compute().item():.4f}")
print(f"Mean Dice Score on test set: {mean_dice_metric_test.compute().item():.4f}")
# %%
mean_iou_metric_train = MeanIoU(num_classes=len(class_str), input_format='index').to(device=device)
mean_dice_metric_train = DiceScore(num_classes=len(class_str), average='macro', input_format='index').to(device=device)

n_sample = 500
indices = np.random.choice(len(dataset_train), size=n_sample, replace=False)

for i in indices:
    img, mask = dataset_train[int(i.item())]
    img = img.to(device)
    mask = mask.to(device)
    input_image = img.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output_mask = model.inference(input_image, sliding_window_size=224, device=device)
    print(output_mask.device, mask.device)
    mean_iou_metric_train.update(output_mask.long(), mask.unsqueeze(0).long())
    mean_dice_metric_train.update(output_mask.long(), mask.unsqueeze(0).long())
    
print(f"Mean IoU on train set: {mean_iou_metric_train.compute().item():.4f}")
print(f"Mean Dice Score on train set: {mean_dice_metric_train.compute().item():.4f}")

# %% Compare segmentation results according to the races
breeds = CatDogSegmentation.breeds
perf_by_breed = {}
breed_cardinal = df_test['breed_id'].value_counts().to_dict()
iou_metric_breed = {}



for breed in breeds:
    iou_metric_breed[breed] = MeanIoU(num_classes=len(class_str), input_format='index').to(device)

for i in range(len(dataset_test)):
    img, mask = dataset_test[i]
    img = img.to(device)
    mask = mask.to(device)
    input_image = img.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output_mask = model.inference(input_image, sliding_window_size=224, device=device)
    breed_id = df_test.loc[i, 'breed_id']
    breed_name = breeds[breed_id]
    iou_metric_breed[breed_name].update(output_mask.long(), mask.unsqueeze(0).long())

for breed in breeds:
    perf_by_breed[breed] = iou_metric_breed[breed].compute().item()
    # print on sorted order
for breed, iou in sorted(perf_by_breed.items(), key=lambda x: x[1], reverse=True):
    print(f"Breed: {breed}, Mean IoU: {iou:.4f}, Number of samples: {breed_cardinal[breeds.index(breed)]}")
    
# %%
# Remark that the segmentation performance are high for shiba inu, saint bernard and newfoundland
# but low for chihuahua, Ragdoll, Persian, americal_bulldog, pomeranian

high_perf_breeds = ['shiba_inu', 'saint_bernard', 'newfoundland']
low_perf_breeds = ['chihuahua', 'Ragdoll', 'Persian', 'american_bulldog', 'pomeranian']

for breed in high_perf_breeds + low_perf_breeds:
    idxs = df_test.index[df_test['breed_id'] == breeds.index(breed)].tolist()
    for i in idxs:
        img, mask = dataset_test[i]
        img = img.to(device)
        mask = mask.to(device)
        input_image = img.unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            output_mask = model.inference(input_image, sliding_window_size=224, device=device)
        show_images([img.cpu(), mask.cpu(), output_mask.squeeze(0).cpu()], 
                    title=[f'Input Image - {breed}', 'Ground Truth Mask', 'Predicted Mask'])
# %%
# %% Compare segmentation results according to the cat or dog
device = "cuda"
model.to(device)
species = CatDogSegmentation.species
perf_by_species = {}
species_cardinal = df_test['label'].value_counts().to_dict()
iou_metric_species = {}


for specie in species:
    iou_metric_species[specie] = MeanIoU(num_classes=3, input_format='index').to(device)

for i in range(len(dataset_test)):
    img, mask = dataset_test[i]
    img = img.to(device)
    mask = mask.to(device)
    input_image = img.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output_mask = model.inference(input_image, sliding_window_size=224, device=device)
    species_id = df_test.loc[i, 'label']
    species_name = species[species_id]
    iou_metric_species[species_name].update(output_mask.long(), mask.unsqueeze(0).long())

for species in species:
    perf_by_species[species] = iou_metric_species[species].compute().item()
    # print on sorted order
for species, iou in sorted(perf_by_species.items(), key=lambda x: x[1], reverse=True):
    species_id = species.index(species)
    print(f"Species: {species}, Mean IoU: {iou:.4f}, Number of samples: {species_cardinal[species_id]}")    
# %%
