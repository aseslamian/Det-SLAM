# Det-SLAM
Det-SLAM is a visual SLAM system that is robust in dynamic scenarios for RGB-D inputs. Using Detctron2 for semantic segmentation and moving object detection.

## How to run the code 
1- you need to install ORB-SLAM3 on ubuntu (I highly recommend to install ubuntu 18.04):
  - first, you need to install some prerequisites like: Eighen3 / Pangolin / DBoW2 / Opencv (the code is sensible to the version of opencv so recommend to install opencv 3.4.1)
  - second, you need to install Cmake (I recommend Cmake version 3.13.3 or lower. The newer version might not work for this code!)
  - Third, you need to creat a "build" file and install Orb-slam3. there are some nice sources that you can easily search for it.
  - for more information you can see "https://github.com/UZ-SLAMLab/ORB_SLAM3"
  
2- for the Detectron2 
   - you need to install some prequisites, please see "https://github.com/facebookresearch/detectron2"
   - I recommend to use Google Colab in order not to challenge with errors made of prequisites version.
   - after you install prequisites and Detectron2 you need to run it for the TUM datasets. if you use ggogle colab all the process will run automatically. you just need to open Colab and run!
   
3- after you download processed dataset you need to run ORB-SLAM3 with the dataset by run command below: 


  `./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml <the place of your dataset>`
 
 
4- Then, the code provide you a trajectory bag so, you can evaluate it with the groundthruth. To do this, you just need to run the python code in the "evaluation" folder.

    python evaluate_ate_scale.py --verbose --scale 1.2 <Groundtruth path> <your result path>
     
 for instance:
        ```python evaluate_ate_scale.py --verbose --scale 1.2 /home/ali/Final/Det-SLAM/Results/f_w_static/groundtruth.txt /home/ali/Final/Det-SLAM/Results/f_w_static/f_w_static_test.txt ```

The code results of the paper are provided. you can check it for practice!

5- Also, you use online platform for evaluation:  " https://vision.in.tum.de/data/datasets/rgbd-dataset/online_evaluation "
      

Good luck
