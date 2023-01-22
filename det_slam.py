
# Det-SLAM: A semantic visual SLAM for highly dynamic scene using Detectron2
# paper: https://arxiv.org/abs/2210.00278

### Part A) Semantic Segmentation

step-1) Import Detectron2 library in colab in order to semantic segmentation of 
moving Objects. We use pre-train weights  "COCO-InstanceSegmentation/
mask_rcnn_X_101_32x8d_FPN_3xl" to configure system for selecting the most known moving objects in datasets.
"""

!python -m pip install pyyaml==5.1
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os, json, cv2, random 
from google.colab.patches import cv2_imshow
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
## Configur system
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

"""step-2) Use the configure system for removing objects. In each picture moving object is recognized and covered with a black mask. So selected part can not be considered in furthur processing of ORB-SLAM3"""

import cv2
import os
from PIL import Image

def load_images_from_folder(folder,folder2):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        outputs = predictor(img)
        masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))
        label = np.array(outputs["instances"].pred_classes.to("cpu"))
        for i in range(masks.shape[0]):
            if label[i] == 0  or label[i] == 56:
                    item_mask = masks[i]
                    cropped_mask = Image.fromarray((item_mask * 255).astype('uint8'))
                    p = np.array(cropped_mask)
                    p.shape = p.shape + (1,)
                    p = (255- p)/255
                    img = np.multiply(p,img)
   
        cv2.imwrite(os.path.join(folder2,filename), img)
    return outputs

"""step-3) Download TUM RGB-D datasets from the source and execute detectron2 on them."""

import tarfile 
print("press the number of input datasets: \n1)walking_static\n2)siting_static\n3)walking_xyz\n4)walking_rpy\n5)walking_half\n  (example:4)")
num = int(input())
if num <1 or num>5:
    print("Error: (number should be between 1 to 5)")
elif num == 1:
     !wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz
     file = tarfile.open('rgbd_dataset_freiburg3_walking_static.tgz')
     file.extractall('/content/') 
     file.close()
elif num == 2:
     !wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz
     file = tarfile.open('rgbd_dataset_freiburg3_sitting_static.tgz')
     file.extractall('/content/') 
     file.close()
elif num == 3:
     !wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
     file = tarfile.open('rgbd_dataset_freiburg3_walking_xyz.tgz')
     file.extractall('/content/') 
     file.close()
elif num == 4:
      !wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz
      file = tarfile.open('rgbd_dataset_freiburg3_walking_rpy.tgz')
      file.extractall('/content/') 
      file.close()
else:
      !wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz
      file = tarfile.open('rgbd_dataset_freiburg3_walking_halfsphere.tgz')
      file.extractall('/content/') 
      file.close()

file = str(file.name.replace('.tgz','/rgb'))
load_images_from_folder(file , file)

"""Now RGB-D datasets are processed by the Detectron2.

### Part B) Depth Processing

Now let's start **Depth Processing**:
"""

import numpy
import array as ay
## Convert Depth Image from RGB to Grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
###
def load_depthIMG_from_folder(folder,folder2):
    depthImg=[]
    for filename in os.listdir(folder):
        depthImg = cv2.imread(os.path.join(folder,filename))
        outputs = predictor(depthImg)
        depthImg = rgb2gray(depthImg)
        masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))
        bbox  = np.array(outputs["instances"].pred_boxes.to("cpu"))
        label = np.array(outputs["instances"].pred_classes.to("cpu"))

        for i in range(masks.shape[0]):
                if label[i] == 0 :
                      item_mask = masks[i]  
                      cropped_mask = Image.fromarray((item_mask * 1).astype('uint16'))
                      objMask = np.array(cropped_mask)
                      pic = depthImg.copy()
                      pic= np.multiply(objMask,pic)
                      MinObj = np.amin(np.array(pic)[pic != 0])     # np.amin(np.array(pic)[pic != np.amin(pic)])
                      MaxObj = np.amax(pic)

                      box = np.array(bbox[i].to("cpu"))  
                      start_point = (round(box[0]),round(box[1]))
                      end_point = (round(box[2]),round(box[3]))
                      [height,width] = depthImg.shape
                      ROImask = np.zeros((height,width), np.uint16)
                      cv2.rectangle(ROImask, start_point,end_point,(1,1),-1);
                      pic = depthImg.copy()
                      pic = np.multiply(ROImask,pic)
                      MinROI = np.amin(np.array(pic)[pic != 0])     # np.amin(np.array(pic)[pic != np.amin(pic)])   
                      MaxROI = np.amax(pic)

                      alpha = 0.001   ## Should be determined
                      d = np.subtract(MaxROI, MaxObj)
                      alpha_d= np.dot(alpha, d)
                      Min_obj2 = np.subtract(MinObj,alpha_d)
                      Max_obj2 = np.add(MaxObj, alpha_d)
                      for m in range(height):
                          for n in range(width):
                              if pic[m,n]>= Min_obj2  and  pic[m,n]<=Max_obj2:
                                    depthImg[m,n]=0

                     
        depthImg = Image.fromarray(depthImg.astype('uint16'))
        depthImg = np.array(depthImg)
        cv2.imwrite(os.path.join(folder2,filename), depthImg)

file = str(file.name.replace('.tgz','/depth'))
load_depthIMG_from_folder(file , file)

"""## Download The Processed Dataset """

path = '/content/rgbd_dataset_freiburg3_walking_halfsphere.zip'
ZipFile  = os.path.join(path,'.zip')
files.download(ZipFile)

import shutil
from google.colab import files
shutil.make_archive("/content/rgbd_dataset_freiburg3_walking_halfsphere", 'zip', "/content/rgbd_dataset_freiburg3_walking_halfsphere")
files.download(ZipFile)

"""Now You can use these data for ORB-SLAM3 in order to mitigate moving objects"""