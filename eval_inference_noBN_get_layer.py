# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:38:25 2019

@author: Ray
"""

import sys
import argparse
from yolo_bp import YOLO, detect_video
from PIL import Image
import os
import pascalvoc_eval
import numpy as np
import matplotlib.pyplot as plt


def detect_img(yolo,image):
    try:
        image = Image.open(image)
    except:
        print('Open Error! Try again!')
    else:
        r_image, label_bbox, layers_feature_list = yolo.get_layers_feature(image)
        #r_image.show()
    #yolo.close_session()
    return r_image, label_bbox, layers_feature_list
    
    
    
val_filename = './val_dehze_kr.txt'

def eval_inference(val_filename):
    model = YOLO()
    with open(val_filename) as f:
        val_lines = f.readlines()

        num_val = len(val_lines)

        # Train data
        train_data = val_lines
        train_images = []
        for data in train_data:
            val_bboxes = []
            image, *bboxes = data.split()
            file_id = os.path.split(image)[-1].split('.')[0]
            train_images.append([image,file_id])

        #class_name_dic = {'shrimp':0,'fodder':1}
    
    total_layers_feature = []
    
    #label_bbox = [label, score, xmin, ymin, xmax, ymax]  
    for i in range(num_val):
        filename = val_lines[i].split()[0].split('/')[-1]
        print("Processing Number:",i,"/",num_val, '###', filename)

        _, label_bbox, layers_feature_list = detect_img(model, train_images[i][0])
        filename_pred_result =  'detections_val_dehze/' + str(train_images[i][1]) + '.txt'
        fp3 = open(filename_pred_result, 'w')
    
        for j in range(len(label_bbox)):
            k_list = str(label_bbox[j][0]) + ' ' +str(label_bbox[j][1]) + ' ' +str(label_bbox[j][2]) + ' ' +str(label_bbox[j][3]) + ' ' +str(label_bbox[j][4]) + ' '+str(label_bbox[j][5])
            fp3.writelines(k_list+'\n')
        fp3.close()
        total_layers_feature.append(layers_feature_list)
    return total_layers_feature

if __name__ == "__main__":
    total_layers_feature = eval_inference(val_filename)
    pascalvoc_eval.pascalvoc_eval()

    #total_layers_feature[0][2].shape
    #len(total_layers_feature[0])
    #layers_feature_list = [ image_data, conv2d_1, leaky_re_lu_1, conv2d_2, leaky_re_lu_2, conv2d_3, leaky_re_lu_3, conv2d_4, leaky_re_lu_4,
    #                           conv2d_5, leaky_re_lu_5, conv2d_6, leaky_re_lu_6, conv2d_7, leaky_re_lu_7, conv2d_8, leaky_re_lu_8, conv2d_9, leaky_re_lu_9,
    #                           conv2d_10, conv2d_11, leaky_re_lu_10, conv2d_12, leaky_re_lu_11, conv2d_13 ]
        
    max_list = []
    min_list = []
    for i in range(135):
        max_list.append(np.max(total_layers_feature[i][24]))
        min_list.append(np.min(total_layers_feature[i][24]))
    print(max(max_list))
    print(min(min_list))
    
    #a = np.reshape(total_layers_feature[0][1],-1)
    #plt.hist(a, bins=140*10)
    #plt.show()