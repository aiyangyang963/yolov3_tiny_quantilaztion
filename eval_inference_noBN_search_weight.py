# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:38:25 2019

@author: Ray
"""

import sys
import argparse
from yolo_serach_weight import YOLO, detect_video
from PIL import Image
import os
import pascalvoc_eval
import numpy as np
import matplotlib.pyplot as plt


def detect_img(yolo,image, orgweight):
    try:
        image = Image.open(image)
    except:
        print('Open Error! Try again!')
    else:
        r_image, label_bbox, getconfig, getweight = yolo.search_layers_weight(image, orgweight)
        #r_image.show()
    #yolo.close_session()
    return r_image, label_bbox, getconfig, getweight
    
    
    
val_filename = './val_dehze_kr.txt'

def eval_inference(model, val_filename, orgweight):
    #model = YOLO()
    #model.set_model_parameter(Pbit, Nbit, 1)

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
        #print("Processing Number:",i,"/",num_val, '###', filename)

        _, label_bbox, getconfig, getweight = detect_img(model, train_images[i][0], orgweight)
        filename_pred_result =  'detections_val_dehze/' + str(train_images[i][1]) + '.txt'
        fp3 = open(filename_pred_result, 'w')
    
        for j in range(len(label_bbox)):
            k_list = str(label_bbox[j][0]) + ' ' +str(label_bbox[j][1]) + ' ' +str(label_bbox[j][2]) + ' ' +str(label_bbox[j][3]) + ' ' +str(label_bbox[j][4]) + ' '+str(label_bbox[j][5])
            fp3.writelines(k_list+'\n')
        fp3.close()
    #model.close_session()
    return getconfig, getweight
    

if __name__ == "__main__":
    filename_0 = 'model_data/quant_weight_log.txt'
    fp0 = open(filename_0, 'w')
    model = YOLO()
    model.set_model_parameter(0, 0, 0)
    orgconfig, orgweight = model.get_weights()
    total_map = []
    now_cwd = os.getcwd()
        
    #exploit the distributed of weights per layer to intitalize the start positive bits 
    init_pbit = [4, 0, 0, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0]
    valid_layers = [0]*13
    p_bit_layers = [0]*13
    n_bit_layers = [0]*13
    best_fodder_AP = 53.22
    best_shrimp_AP = 70.62
    w_AP_tmp = 0
    for layer in range(13): # layers = 13
        layer_map = []
        flag = 0    #shrimp & fodder mAP > original , early finish
        #for bw in range(0,15,1): # bitwidth = 1~16bit
        for p in range(4):
            #for p in range(4): #search the decimal place (小數點位置) ,  >> 0~3
            for bw in range(0,15,1):    
                if (flag==0):
                    p_bit = init_pbit[layer] - p 
                    n_bit = 15 - p_bit - bw 
                    
                    
                    #model.p_bit = p_bit
                    #model.n_bit = n_bit
                    p_bit_layers[layer] = p_bit
                    n_bit_layers[layer] = n_bit
                    valid_layers[layer] = 1
                    #model.valid = valid_layer
                    model.set_pnbit_valid(p_bit_layers, n_bit_layers, valid_layers)
                    getconfig, getweight = eval_inference(model, val_filename, orgweight)
        
                    AP_eval, mAP = pascalvoc_eval.pascalvoc_eval()
                    layer_map.append([p_bit, n_bit, AP_eval, mAP])
                    os.chdir(now_cwd)
                    fodder_AP = float(AP_eval[0][1][:-1])
                    shrimp_AP = float(AP_eval[1][1][:-1])
                    w_AP = (fodder_AP + shrimp_AP*1.5)/2
                    if(w_AP>w_AP_tmp):
                        w_AP_tmp = w_AP
                        p_bit_layers[layer] = p_bit
                        n_bit_layers[layer] = n_bit
                    
                    if((fodder_AP>=best_fodder_AP)&(shrimp_AP>=best_shrimp_AP)):
                        flag=1
                        best_fodder_AP = float(AP_eval[0][1][:-1])
                        best_shrimp_AP = float(AP_eval[1][1][:-1])
                        p_bit_layers[layer] = p_bit
                        n_bit_layers[layer] = n_bit
                    lines = 'Layer:' + str(layer) + ' ,P_bit:' + str(p_bit) + ' ,N_bit:' + str(n_bit) + '\n'
                    lines += 'fodder_AP:'+ str(fodder_AP) + 'shrimp_AP:' + str(shrimp_AP) + 'map:' + str(mAP) + 'wAP_tmp:' + str(w_AP_tmp) + '\n'
                    fp0.write(lines)
                    print(lines)
                    print(model.p_bit_list)
                    print(model.n_bit_list)
                    print(model.valid_list)
        total_map.append(layer_map)
        model.p_bit = p_bit_layers
        model.n_bit = n_bit_layers
        
    filename_in = 'model_data/quant_weight_result.txt'
    fp1 = open(filename_in, 'w')

       
    for i in range(13):
        line = 'Layer' + str(i) + ': P_bit' + str(p_bit_layers[i]) + ' N_bit' + str(n_bit_layers[i])
        print(line)
        fp1.write(line+'\n')
    fp1.close()    
        
            
        
        
        
#
'''
tmp = np.array([0.8,-0.8,0.9,-1,0.4,-0.4,0])
tmp = getweight[6][0]
tmp2 = quant_weight(tmp, 1, 3, 1)        
'''