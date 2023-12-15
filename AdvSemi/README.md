# Project name
AdvSemi-CRF
# Title of manuscript
Towards identification if coal macerals through semi-supervised semantic segmentation combined with conditional random fields algorithm

# Author details


## Prerequisite

* CUDA/CUDNN
* pytorch >= 0.2 (We only support 0.4 for evaluation. Will migrate the code to 0.4 soon.)
* python-opencv >=3.4.0 (3.3 will cause extra GPU memory on multithread data loader)


## Installation

* Clone this repo

```bash
git clone https://github.com/hfslyc/AdvSemiSeg.git
```

* Place VOC2012 dataset in `AdvSemiSeg/dataset/VOC2012`. For training, you will need the augmented labels ([Download](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip)). The folder structure should be like:
```
AdvSemiSeg/dataset/VOC2012/JPEGImages
                          /SegmentationClassAug
```

## Testing on VOC2012 validation set with pretrained models

```
python evaluate_voc.py --pretrained-model semi0.125 --save-dir results
```

It will download the pretrained model with 1/8 training data and evaluate on the VOC2012 val set. The colorized images will be saved in ``results/`` and the detailed class IOU will be saved in ``results/result.txt``. The mean IOU should be around ``68.8%``.

* Available ``--pretrained-model`` options: ``semi0.125``, ``semi0.25``, ``semi0.5`` , ``advFull``. 


## Training on VOC2012

```
python train.py --snapshot-dir snapshots \
                --partial-data 0.125 \
                --num-steps 20000 \
                --lambda-adv-pred 0.01 \
                --lambda-semi 0.1 --semi-start 5000 --mask-T 0.2
```

The parameters correspond to those in Table 5 of the paper.

To evaluate trained model, execute the following:

```
python evaluate_voc.py --restore-from snapshots/VOC_20000.pth \
                       --save-dir results
```
