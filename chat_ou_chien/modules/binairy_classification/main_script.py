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
num_class = 2
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_class)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
model.train()

df = create_df(cls_list_path=os.path.join(project_abs_dir,"data/annotations/trainval.txt"),
               image_path=os.path.join(project_abs_dir, "data/images"))
catdog = CatDogBinary(df, transforms=v2.Resize((224,224)))

criterion = torch.nn.CrossEntropyLoss()
preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()

#%%
indices = np.random.randint(low=0, high=1000, size=(5,))


img_samples, species_samples = catdog[indices]

species_samples= torch.tensor(species_samples)
input = torch.stack(img_samples)

input_transformed = preprocess(input)

out = model(input_transformed)
loss = criterion(out, species_samples)
print(loss)
#%%

# 5. Get class label
classes =  EfficientNet_B0_Weights.DEFAULT.meta["categories"]
pred_idx = out.argmax(dim=1)
plot(img_samples, titles=[classes[id] for id in pred_idx])
# %%
