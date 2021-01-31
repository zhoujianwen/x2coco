# -*- coding: utf-8 -*-
'''
@time: 2020/10/28 12:06
spytensor,zhoujianwen
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import time
from datetime import datetime
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split

np.random.seed(41)

# 0为背景
classname_to_id = {"__background__": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


class Txt2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(float(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(os.path.join(self.image_dir, path))
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        # label = shape[-1]
        points = shape[:8]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [self._get_seg(points)]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):  # 需要额外处理by zhoujianwen
        min_x = min_y = np.inf
        max_x = max_y = 0
        arraypoints = np.array(points).reshape(4,2)
        for x, y in arraypoints:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # Segmentation的格式：[x1,y1,x1,y2,x2,y2,x1,y2]
    def _get_seg(self, points):
        x1, y1, x2, y2, x3, y3, x4, y4 = points
        return [x1, y1, x2, y2, x3, y3, x4, y4]


if __name__ == '__main__':
    txt_file = ".\\rawdata\\labels"
    image_dir = ".\\rawdata\\images"
    saved_coco_path = "convertedData"
    saved_time = format(datetime.now(),"%Y%m%d%H%M%S")
    # 整合txt格式标注文件
    total_txt_annotations = {}
    annotations = []

    files = os.listdir(txt_file)
    files.sort()

    # 获取images目录下所有的txt文件列表
    txt_list_path = glob.glob(os.path.join(txt_file,"*.txt"))
    for f in txt_list_path:
        fh1 = open(f, "r", encoding="UTF-8")
        nameOfImage = f.replace("txt", "tif")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            x1 = float(splitLine[1])
            y1 = float(splitLine[2])
            x2 = float(splitLine[3])
            y2 = float(splitLine[4])
            x3 = float(splitLine[5])
            y3 = float(splitLine[6])
            x4 = float(splitLine[7])
            y4 = float(splitLine[8])
            one_box = [nameOfImage, x1, y1, x2, y2, x3, y3, x4, y4, cls]
            annotations.append(one_box)
        fh1.close()

    # annotations = pd.read_csv(csv_file, header=None).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_txt_annotations.keys():
            total_txt_annotations[key] = np.concatenate((total_txt_annotations[key], value), axis=0)
        else:
            total_txt_annotations[key] = value
    # 按照键值划分数据
    total_keys = list(total_txt_annotations.keys())

    train_keys, test_keys = train_test_split(total_keys, test_size=0.2)
    # 如果你不想划分数据集，请把下面两行代码的注释#去掉
    # train_keys = total_keys
    # test_keys = []
    print("train_n:", len(train_keys), 'test_n:', len(test_keys))
    # 创建必须的文件夹
    if not os.path.exists('%s/coco/annotations/' % saved_coco_path):
        os.makedirs('%s/coco/annotations/' % saved_coco_path)
    if not os.path.exists('%s/coco/images/train%s/' %(saved_coco_path,saved_time)):
        os.makedirs('%s/coco/images/train%s/' %(saved_coco_path,saved_time))
    if not os.path.exists('%s/coco/images/test%s/' %(saved_coco_path,saved_time)):
        os.makedirs('%s/coco/images/test%s/' %(saved_coco_path,saved_time))
    # 把训练集转化为COCO的json格式
    l2c_train = Txt2CoCo(image_dir=image_dir, total_annos=total_txt_annotations)
    train_instance = l2c_train.to_coco(train_keys)
    l2c_train.save_coco_json(train_instance, '%s/coco/annotations/instances_train%s.json' %(saved_coco_path,saved_time))
    for file in train_keys:
        shutil.copy(os.path.join(image_dir,file), "%s/coco/images/train%s/" %(saved_coco_path,saved_time))
    for file in test_keys:
        shutil.copy(os.path.join(image_dir,file), "%s/coco/images/test%s/" %(saved_coco_path,saved_time))
    # 把验证集转化为COCO的json格式
    l2c_val = Txt2CoCo(image_dir=image_dir, total_annos=total_txt_annotations)
    val_instance = l2c_val.to_coco(test_keys)
    l2c_val.save_coco_json(val_instance, '%s/coco/annotations/instances_test%s.json' %(saved_coco_path,saved_time))
