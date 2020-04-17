from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
from tqdm import tqdm
import time

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import matplotlib.pyplot as plt
# %matplotlib inline

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """이미지 디렉토리에서 인풋 데이터를 찾아 데이터로 변환한다"""
    
    ## image_dir가 존재하지 않는다면 오류 출력
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    
    result = {}
    
    ### image_dir 내 하위 디렉토리(label)를 가져온다 
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
            
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        
        ## 파일이 없거나 데이터가 작으면 예외 처리
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print("WARNING: Folder has less than 20 images, which may cause issues.")
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print("WARNING: Folder {} has more than {} images. Some images will never be selected".format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        
        ## 트레이닝 / 밸리데이션 / 테스트셋으로 나눈다.
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                               (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                              (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
        
    return result

## 데이터를 다운로드받을 때 사용할 Tqdm 클래스를 정의한다.
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)