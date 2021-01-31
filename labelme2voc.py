import os
import numpy as np
import codecs
import json
from datetime import datetime
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
#1.标签路径
saved_time = format(datetime.now(), "%Y%m%d%H%M%S")
labelme_path = ".\\rawdata\\labelme"              #原始labelme标注数据路径
saved_path = (".\\convertedData\\voc\\%s" % saved_time)                #保存路径


#2.创建要求文件夹
if not os.path.exists("%s/Annotations" % saved_path):
    os.makedirs("%s/Annotations" % saved_path)
if not os.path.exists("%s/JPEGImages" % saved_path):
    os.makedirs("%s/JPEGImages" % saved_path)
if not os.path.exists("%s/ImageSets" % saved_path):
    os.makedirs("%s/ImageSets" % saved_path)
    
#3.获取待处理文件
files = glob("%s/*.json" % labelme_path)
#files = [i.split("/")[-1].split(".json")[0] for i in files]

#4.读取标注信息并写入 xml
for json_file in files:
    #json_file_path = ("%s.json" % json_file)
    (filepath,fullname) = os.path.split(json_file)
    (filename,extension) = os.path.splitext(fullname)
    json_file_details = json.load(open(json_file,"r",encoding="utf-8"))
    height, width, channels = cv2.imread("%s/%s.jpg" %(filepath,filename)).shape
    with codecs.open(("%s/Annotations/%s.xml" %(saved_path,filename)),"w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + filename + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>ChaojieZhu</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file_details["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:,0])
            xmax = max(points[:,0])
            ymin = min(points[:,1])
            ymax = max(points[:,1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>'+label+'</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
        print(fullname,xmin,ymin,xmax,ymax,label)
        xml.write('</annotation>')
        
#5.复制图片到JPEGImages目录下
image_files = glob(os.path.join(labelme_path,"*.jpg"))
print(image_files)
print("copy image files to JPEGImages")
for image in image_files:
    shutil.copy(image,"%s/JPEGImages" % saved_path)
    
#6.split files for txt
# 打开并创建必须的文件
txtsavepath = "%s/ImageSets" % saved_path
ftrainval = open(("%s/trainval.txt" % txtsavepath), 'w')
#ftest = open(("%s/test.txt" % txtsavepath), 'w')
ftrain = open(("%s/train.txt" % txtsavepath), 'w')
fval = open(("%s/val.txt" % txtsavepath), 'w')
total_files = glob(("%s/Annotations/*.xml" % saved_path))
#total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    (filepath,fullname) = os.path.split(file)
    ftrainval.write(fullname + "\n")
#test
#for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
#split
train_files,val_files = train_test_split(total_files,test_size=0.2,random_state=20)
#train
for file in train_files:
    (filepath, fullname) = os.path.split(file)
    ftrain.write(fullname + "\n")
#val
for file in val_files:
    (filepath, fullname) = os.path.split(file)
    fval.write(fullname + "\n")

ftrainval.close()
ftrain.close()
fval.close()
#ftest.close()
