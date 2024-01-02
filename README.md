# Project name
AdvSemi-CRF
# Title of manuscript
Towards identification if coal macerals through semi-supervised semantic segmentation combined with conditional random fields algorithm

# Author details
* Na Xu: xuna1011@gmail.com; xuna@cumtb.edu.cn
* Qingfeng Wang: wangqf0716@gmail.com

# Guidance
# 1. semi-supervised semantic segmentation

## Prerequisite
* CUDA/CUDNN
* pytorch >= 0.2 (We only support 0.4 for evaluation. Will migrate the code to 0.4 soon.)
* python-opencv >=3.4.0 (3.3 will cause extra GPU memory on multithread data loader)

## Data availability
* Data will be made available on request

## Installation

* Clone this repo
* Place dataset(contain images and labels) in `AdvSemi/dataset/`. The folder structure should be like:
```
AdvSemi/dataset/JPEGImages
                  /SegmentationClassAug
```

## Training

```
python train.py --snapshot-dir snapshots \
                --partial-data 0.25 \
                --num-steps 20000 \
                --lambda-adv-pred 0.01 \
                --lambda-semi 0.1 --semi-start 5000 --mask-T 0.2
```
To evaluate trained model, execute the following:

```
python evaluate_voc.py --restore-from snapshots/VOC_20000.pth \
                       --save-dir results
```

## Testing
* Available ``--pretrained-model`` options: ``semi0.125``, ``semi0.25``, ``semi0.5`` , ``advFull``.
It will download the pretrained model with 1/4 training data and evaluate on the testing set. The colorized images will be saved in ``results/`` .

# 2. conditional random fields algorithm

## dataset
* Save the prediction results folder as input to the algorithm
* One-to-one correspondence with labels

## Run and display
* Run files with the `main.py` in CRF\ folders.
* The results will be saved locally.

## Evaluate
* Run files with the `evaluate.py` in CRF\ folders.

