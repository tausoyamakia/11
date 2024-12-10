import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import model

modle = model.MyModle
print(model)

ds_train = datasets.FanshionMNIST(
    root='data',
    train=True ,
    download=True,
    transform=transforms.Compse([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
)

image, target = ds_train[0]
image = image.unsqueeze(dim=0)

model.eval()
with torch.no_grad():
    logits = model(image)
    
print(logits)