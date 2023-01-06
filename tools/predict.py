
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import _init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str,
                        default='cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='model_best.pth.tar')

    args = parser.parse_args()
    update_config(config, args)
    return args

def main():
    args = parse_args()
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()



    def transform_image(image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)


    model.eval()
 
    def get_prediction(image_bytes):
        tensor = transform_image(image_bytes=image_bytes)
        cuda0 = torch.device('cuda:0')
        tensor=tensor.to(cuda0)
        output = model.forward(tensor)
     
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, classes = torch.max(probs, 1)
        index_to_breed = {0: 'Benign40X', 1: 'G340X', 2: 'G440X', 3: 'G540X'}
        return conf.item(), index_to_breed[classes.item()]
 
    #image_path=input("Enter the image path:")
    image_path=r"templates\images\1.tif"
    image = plt.imread(image_path)
    plt.imshow(image)
    
 
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
 
        conf,y_pre=get_prediction(image_bytes=image_bytes)
        if y_pre == 'Benign40X':
            print("You have no Cancer (Confidence score : {0:.2f})".format(conf))

        if y_pre == 'G340X':
            print('You have a Grade 3 Cancer (Confidence score : {0:.2f})'.format(conf))
        #print(y_pre, ' at confidence score : {0:.2f}'.format(conf))

        if y_pre == 'G440X':
            print('You have a Grade 4 cancer (Confidence score : {0:.2f})'.format(conf))

        if y_pre == 'G540X':
            print('You have a Grade 5 cancer (Confidence score : {0:.2f})'.format(conf))
        #plt.title(y_pre + ' at confidence score : {0:.2f}'.format(conf))
        #plt.show()


if __name__ == '__main__':
    main()
    #models.predict()
    
