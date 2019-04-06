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
        r_image, label_bbox, conv2d_1 = yolo.detect_image_box(image)
        #r_image.show()
    #yolo.close_session()
    return r_image, label_bbox, conv2d_1
    
    
    
val_filename = './val_dehze_kr.txt'

def eval_inference(val_filename, Pbit, Nbit):
    model = YOLO()
    model.set_model_parameter(Pbit, Nbit,1)

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
    
    
    #label_bbox = [label, score, xmin, ymin, xmax, ymax]  
    for i in range(num_val):
        filename = val_lines[i].split()[0].split('/')[-1]
        print("Processing Number:",i,"/",num_val, '###', filename)

        _, label_bbox, conv2d_1 = detect_img(model, train_images[i][0])
        filename_pred_result =  'detections_val_dehze/' + str(train_images[i][1]) + '.txt'
        fp3 = open(filename_pred_result, 'w')
    
        for j in range(len(label_bbox)):
            k_list = str(label_bbox[j][0]) + ' ' +str(label_bbox[j][1]) + ' ' +str(label_bbox[j][2]) + ' ' +str(label_bbox[j][3]) + ' ' +str(label_bbox[j][4]) + ' '+str(label_bbox[j][5])
            fp3.writelines(k_list+'\n')
        fp3.close()
    #model.close_session()
    return conv2d_1
    

if __name__ == "__main__":
    
    total_map = []
    now_cwd = os.getcwd()
    for i in range(1):
        
        p_bit = 6 - i
        n_bit = 15 - p_bit
    
        conv2d_1 = eval_inference(val_filename,p_bit, n_bit)
        AP_eval, mAP = pascalvoc_eval.pascalvoc_eval()
        total_map.append([p_bit,AP_eval, mAP, conv2d_1])
        os.chdir(now_cwd)