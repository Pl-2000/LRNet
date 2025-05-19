# LRNet
LRNet: Change Detection of High-Resolution Remote Sensing Imagery via Strategy of Localization-then-Refinement

## LRNet: Change Detection of High-Resolution Remote Sensing Imagery via Strategy of Localization-then-Refinement
To address edge discrimination challenges in change detection, a novel network based on a localization-then-refinement strategy is proposed in this paper, namely LRNet. LRNet consists of two stages: localization and refinement. In the localization stage, a three-branch encoder simultaneously extracts original image features and their differential features for interactive localization of change areas. To minimize information loss during feature extraction, learnable optimal pooling (LOP) is proposed as a trainable alternative to max-pooling, enhancing overall network optimization. Additionally, change alignment attention (C2A) and hierarchical change alignment module (HCA) are designed to effectively interact multi-branch features and accurately locate change areas of varying sizes. In the refinement stage, the edge-area alignment module (E2A) refines localization results by constraining change areas and edges. The decoder, combined with the difference features strengthened by C2A from the localization phase, further refines changes across different scales, achieving precise boundary discrimination. The proposed LRNet outperforms 13 other state-of-the-art methods on comprehensive metrics and delivers the most accurate boundary discrimination results on the LEVIR-CD and WHU-CD datasets.

## How to use the code
###  Environment configuration 
Deep learning framework: Pytorch

### Code configuration
1. main_LRNet_IOU.py is the main program.   

## Dataset
Two publicly available CD datasets are used to evaluate the model's performance: the LEVIR-CD and WHU-CD datasets.
