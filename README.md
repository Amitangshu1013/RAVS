# RAVS
- The complete code setup will be available soon. 

-   Please follow [Torch Attacks](https://github.com/Harry24k/adversarial-attacks-pytorch) and [Glance and Focus Networks](https://github.com/blackfeather-wang/GFNet-Pytorch) to set up virtual environment or miniconda environment. The entire code base is in PyTorch. Please follow the recommended versions from Torch Attacks. This should work for GFNet as well.​
    
-   Two inference scripts are provided – Passive methods and GFNet. Required comments are added. These are scripts to reproduce results for Table 1 from main paper and Table 1 from supplementary. ​
    
-   Passive Method - Inference script to test passive methods is torch_attack_ImageNet_full.py. ​
    
-   Attack Verification: When the attack parameters are set to 0, the results serve as a baseline indicating no attack. This serves as an additional check to ensure the attack mechanism is functioning correctly. The accuracies for clean samples have been verified under two distinct settings: one with standard dataloader images and the other with all attack parameters set to 0. The accuracies for the base case align with the numbers reported on [PyTorch](https://pytorch.org/vision/master/models.html). ImageNet1K_V1 weights are used since FALcon and GFNet use these weights for their respective evaluations. ​
    
-   Glance and Focus Networks - Inference script to test GFNet is inference_adversarial.py. After git cloning the repository from the respective GitHub page, add this script to the folder. The attack verification for GFNet is also performed. ​
    
-   Checkpoints – The checkpoints are provided on the repository. We have used [GFNet-96](https://drive.google.com/file/d/1Iun8o4o7cQL-7vSwKyNfefOgwb9-o9kD/view) . These models are with respect to the paper “Glance and Focus: a Dynamic Approach to Reducing Spatial Redundancy in Image Classification”, NeurIPS-2020. ​
    
-   Evaluation : Please follow Evaluate Pre-trained models in their README and execute eval mode 1. Just replace inference.py with inference_adversarial.py ​
    
-   [Robustness library](https://github.com/MadryLab/robustness) – Script to test Adversarially trained ResNet50 isn’t provided but will be provided if required. The weights used are on this [link](https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0). ​

|Model| Reproduced - No Attack (eps=0) |
|-|--|
| ResNet34 |73.30  |
| ResNet50 |76.15  |
| Adv ResNet50 |47.91  |
| FALcon ResNet50 |72.97  |
| GFNet ResNet50 |75.88  |

- White Box attack on Adversarially trained ResNet 50 (PGD) as per robustness library 

|Eps|Robustness Library  |
|--|--|
| 0/255 | 47.91 |
| 4/255 | 33.06 |
| 8/255 | 19.63 |



- White Box attack on Adversarially trained ResNet 50 (PGD) as reproduced with TorchAttacks

|Eps|TorchAttacks  |
|--|--|
| 0/255 | 47.91 |
| 4/255 | 33.05 |
| 8/255 | 20.12 |

      
