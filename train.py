import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from PIL import Image
from unet import *
import os
import matplotlib.pyplot as plt

jpg = os.listdir('/content/unet-improvement/semantic_segmentation_unet/data/train_images')[0]
jpg_mask = os.listdir('/content/unet-improvement/semantic_segmentation_unet/data/train_masks')[0]
img_train = Image.open(f"/content/unet-improvement/semantic_segmentation_unet/data/train_images/{jpg}")
img_test = Image.open(f"/content/unet-improvement/semantic_segmentation_unet/data/train_masks/{jpg[:-4] + '_mask.gif'}")


convert_tensor = transforms.ToTensor()
for jpg in os.listdir('images')[:20]:
  x = Image.open(f"images{jpg}")
  y = Image.open(f"images{jpg + '_mask.png'}")
  x = convert_tensor(x).unsqueeze(0).resize_((1,3,224,224))
  y = convert_tensor(y).unsqueeze(0).resize_((1,2,224,224))
  x_train = torch.cat((x_train, x))
  y_train = torch.cat((y_train, y))

x_train = x_train[1:]
y_train = y_train[1:]

encoders = ['unet_encoder','resnet18','resnet34','resnet50','resnet101','resnet152']
for encoder in encoders:
  net = Unet(backbone_name=encoder,classes=1)

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(net.parameters())
  losses = []
  for _ in range(10):
      with torch.set_grad_enabled(True):
          batch = x_train
          targets = y_train

          out = net(batch)
          loss = criterion(out, targets)
          loss.backward()
          optimizer.step()
      losses.append(loss)
  plt.plot(losses,label=encoder)
plt.legend()