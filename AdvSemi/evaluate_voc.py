import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
from packaging import version

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplab import Res_Deeplab
# from dataset.voc_dataset import VOCDataSet
from dataset.my_dataset import VOCDataSet

from PIL import Image

import matplotlib.pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'

DATA_DIRECTORY = './dataset/my_dataset_list/dataset'
DATA_LIST_PATH = './dataset/my_dataset_list/val.txt'
IGNORE_LABEL = 255
# NUM_CLASSES = 21
NUM_CLASSES = 6
NUM_STEPS = 147 # Number of images in the validation set.
# RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-8d75b3f1.pth'
RESTORE_FROM = "./snapshots/VOC_20000.pth"
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'


pretrianed_models_dict ={'semi0.125': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-03c6f81c.pth',
                         'semi0.25': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.25-473f8a14.pth',
                         'semi0.5': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.5-acf6a654.pth',
                         'advFull': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSegVOCFull-92fbc7ee.pth'}


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ1, j_list1, M = ConfM.jaccard()
    aveJ2, j_list2, M = ConfM.accuracy()
    aveJ3, j_list3, M = ConfM.recall()


    
    classes = np.array(('background',  # always index 0
               'JZT', 'LZT', 'DZT','', 
               ''))

    for i, iou in enumerate(j_list1):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list1[i]))

    print('meanIOU: ' + str(aveJ1) + '\n')

    for i, iou in enumerate(j_list2):
        print('class {:2d} {:12} accuracy {:.2f}'.format(i, classes[i], j_list2[i]))
    print('meanAccuracy: ' + str(aveJ2) + '\n')

    for i, iou in enumerate(j_list3):
        print('class {:2d} {:12} recall {:.2f}'.format(i, classes[i], j_list3[i]))
    print('meanRecall: ' + str(aveJ3) + '\n')
        
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list1):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list1[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ1) + '\n')

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    
    classes = np.array(('background',  # always index 0
               'JZT', 'LZT', 'DZT','',''))
    
    colormap = [(0,0,0),(0,0,0.5),(0.5,0,0),(0,0,0.5),(0.5,0.5,0),(0.5,0.5,0.5)]
#    colormap = [(0,0,0),(0,0,0.5),(0.5,0,0),(0,0,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds = [0,1,2,3,4,5]
#    bounds = [0,1,2,3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    model = Res_Deeplab(num_classes=args.num_classes)

    if args.pretrained_model != None:
        args.restore_from = pretrianed_models_dict[args.pretrained_model]

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=False)
    else:
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
    data_list = []

    colorize = VOCColorize()

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda(gpu0))
        output = interp(output).cpu().data[0].numpy()

        output = output[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

       
#        # 保存裁剪后的图像
#        cropped_label_path = os.path.join(args.save_dir, '{}_label.png'.format(name[0]))
#        cropped_label = Image.fromarray(label[1].numpy().transpose(1, 2, 0).astype(np.uint8))
##        cropped_image = cropped_image.crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h))
#        cropped_label.save(cropped_label_path)

        # 保存分割结果
        filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)

        # show_all(gt, output)
        data_list.append([gt.flatten(), output.flatten()])

    filename = os.path.join(args.save_dir, 'result.txt')
    get_iou(data_list, args.num_classes, filename)


if __name__ == '__main__':
    main()
