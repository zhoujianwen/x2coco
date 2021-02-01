#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 0:37
# @Author  : zhoujianwen
# @Email   : zhou_jianwen@qq.com
# @File    : vgg2coco.py
# @Describe: 


# In[ ]:


import json
import os, cv2
import numpy as np
from datetime import datetime

# In[ ]:


def get_structure_properties(shapes):
    x = shapes['all_points_x']
    y = shapes['all_points_y']
    points = []
    contour = []
    for i, val in enumerate(x):
        points.append(val)
        points.append(y[i])
        contour.append([val, y[i]])
    ctr = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
    area = cv2.contourArea(ctr)
    rect = cv2.boundingRect(ctr)
    x, y, w, h = rect
    bbox = [x, y, w, h]

    return points, bbox, area


# In[ ]:


def via_to_coco(infile, outfile, image_path):
    vgg_json = open(infile)
    vgg_json = json.load(vgg_json)

    main_dict = {}
    info = "{'year': 2020, 'version': '1', 'description': 'Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)', 'contributor': '', 'url': 'http://www.robots.ox.ac.uk/~vgg/software/via/', 'date_created': 'Sun Feb 02 2020 11:47:26 GMT+0100 (Central European Standard Time)'}"
    image_list = list(vgg_json.keys())

    images = []

    for i, img in enumerate(vgg_json['_via_img_metadata']):
        image = {}
        #print(vgg_json[img])
        (filepath,fullname) = os.path.split(img)
        (filename,extension) = os.path.splitext(fullname)
        if len(extension) > len(".jpg"):
            im = cv2.imread(os.path.join(image_path,filename+".jpg"))
        else:
            im = cv2.imread(os.path.join(image_path,img))
        h, w, c = im.shape
        image['id'] = i
        image['width'] = w
        image['height'] = h
        image['file_name'] = ("%s.jpg" % filename)
        image['coco_url'] = ("%s.jpg" % filename)
        images.append(image)

    annotations = []
    image_id = 0
    for i, v in enumerate(vgg_json["_via_img_metadata"]):
        (filepath,fullname) = os.path.split(v)
        (filename,extension) = os.path.splitext(fullname)
        # if len(extension) > len(".jpg"):
        #     data = vgg_json["_via_img_metadata"]["%s.jpg" % filename]
        #     print(data)
        # else:
        data = vgg_json["_via_img_metadata"][v]
        regions = data["regions"]
        for j, r in enumerate(regions):
            shape_attributes = r["shape_attributes"]
            region_attributes = r["region_attributes"]
            try:
                # replace the key Objekte with yours
                objekt = region_attributes["antenna"]
            except:
                print('No Object keyword for ', v)
                continue
            segmentation, bbox, area = get_structure_properties(shape_attributes)
            anno = {}
            anno['id'] = image_id
            anno['image_id'] = i
            anno['category_id'] = 0
            anno['segmentation'] = segmentation
            anno['area'] = area
            anno['bbox'] = bbox
            anno['iscrowd'] = 0
            image_id += 1
            annotations.append(anno)

    main_dict['info'] = info
    main_dict['images'] = images
    main_dict['annotations'] = annotations
    main_dict['categories'] = [{"id": 0,"name": "antenna"}]
    (filepath, fullname) = os.path.split(outfile)
    (filename, extension) = os.path.splitext(fullname)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    json.dump(main_dict, open(outfile, 'w',encoding='utf-8'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示
    # with open(outfile, 'w') as f:
    #     json.dump(main_dict, f, ensure_ascii=False, indent=2)  # indent=2 更加美观显示
    #     f.close()


# In[ ]:


if __name__ == '__main__':
    vocFolderName = "202102010034"
    vgg_json_file = "./rawdata/voc/%s/%s" %(vocFolderName, "train2017.json")   # path_to_input_via_json_file
    saved_time = format(datetime.now(), "%Y%m%d%H%M%S")
    cocoFolderName = saved_time
    json_file = ('./convertedData/coco/%s/annotations/instances_train%s.json' % (cocoFolderName, cocoFolderName))  # path_to_output_coco_json_file
    path_images = "./rawdata/voc/202102010034/JPEGImages" # path_to_images
    via_to_coco(vgg_json_file, json_file, path_images)

