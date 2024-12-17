"""
This code provides inference using for passive baseline methods. 
Need to provide path to ImageNet validation set. 
"""

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights
from torchvision.models import vision_transformer, ViT_B_16_Weights
from torchvision.models import swin_transformer, Swin_T_Weights, Swin_B_Weights, Swin_V2_T_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import torch
from torch.utils.data import DataLoader

#CUDA Deterministic set to True to create same adversarial samples with TorchAttacks
import torchattacks
torch.backends.cudnn.deterministic = True
from torchattacks import PGD

import torchvision
import matplotlib.pyplot as plt
import numpy as np


# Transforms for ImageNet as defined by TorchAttacks

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = trn.Compose(
    [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])


# ImageNet-Full
val_dataset = dset.ImageFolder(root="enter your path here", transform=test_transform)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True,
                                         num_workers=4, pin_memory=True)




# Initialize Surrogate and Target Passive Baselines
# net_1 is Surrogate; net_2 is Target


#net_1 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
net_1 = models.resnet34(weights=ResNet34_Weights.DEFAULT)
net_2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#net_1 = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
net_1.eval();
net_2.eval();
net_1.cuda();
net_2.cuda();

# Initialize Attack
atk = PGD(net_1, eps=0, alpha=0, steps=10, random_start=True) # No Attack for base case. 
#atk = PGD(net_1, eps=8/255, alpha=2/255, steps=10, random_start=True)
#atk = torchattacks.MIFGSM(net_1, eps=8/255, alpha=2/255, steps=10, decay=1.0)
#atk = torchattacks.VMIFGSM(net_1, eps=8/255, alpha=2/255, steps=10, decay=1.0)
#atk = torchattacks.PIFGSM(net_1, max_epsilon=8/255, num_iter_set = 10)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Since images are normalized.


# Basic evaluation code
# Evaluate

correct_adv_1 = 0
n_samples = 0
with torch.no_grad():
    for images,labels in val_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.enable_grad():
            adv_images = atk(images, labels)
        #breakpoint()
        adv_images = adv_images.cuda()
      

        pred_adv_1 = net_2(adv_images)
        correct_adv_1 += (pred_adv_1.argmax(1) == labels).sum().item()

        n_samples += adv_images.shape[0]

        #breakpoint()    


robust_accuracy_1 = 100*(correct_adv_1 / n_samples)



print(f"Robust accuracy Transfer: {robust_accuracy_1:>0.2f}%")
print("\n")