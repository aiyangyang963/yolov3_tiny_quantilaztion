# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:39:07 2019

@author: Ray
"""

import h5py 
import numpy as np
f = h5py.File('model_data/yolov3-tiny_obj_final123.h5', 'r')
#f = h5py.File('model_data/yolov3_weights.h5', 'r')

list(f.keys())

#list(f['conv2d_1/conv2d_1/kernel:0'])



layer_name=[]
layer_value=[]

for key in list(f.keys()):
    for i in list(f[key]):
        for j in list(f[key][i]):
            print(key,i,j)
            layer_name.append(key+'/'+i+'/'+j)
            print(list(f[key][i][j]))
            layer_value.append(list(f[key][i][j]))
            
            
print(layer_name[0])
print(layer_value[0])
           
'''================================================================================'''

m = np.zeros_like(conv2d_1)
#
for i in range(16):
    for j in range(512):
        for k in range(512):
            m[0][j][k][i] = ( (conv2d_1[0][j][k][i] - layer_value[2][i])/np.sqrt(layer_value[3][i]+1e-3) ) * layer_value[1][i] + layer_value[0][i]

m[0][0][0][:] - batch_normalization_1[0][0][0][:]
batch_normalization_1[0][0][0][:]




#padding
mm = np.zeros_like(conv2d_1)
in_p = np.zeros([1,514,514,3])

for i in range(3):
    for j in range(512):
        for k in range(512):
            in_p[0][j+1][k+1][i] = input_data1[0][j][k][i]

# conv
mm = np.zeros_like(conv2d_1)

for k in range(16):
    for i in range(512):
        for j in range(512):
            for c in range(3):
                for a in range(3):
                    for b in range(3):
                        mm[0][i][j][k]  += in_p[0][b+i][a+j][c]*getweight[1][0][b][a][c][k]

#convert weight
weight1 = getweight[1]
beta1 = getweight[2][1]
gamma1 = getweight[2][0]
mean1 = getweight[2][2]
var1 = getweight[2][3]


def ConvBN2ConvBias(weight, beta, gamma, mean, var):
    
    new_weight = np.zeros_like(weight)
    new_bias = np.zeros(weight.shape[3])
    

    for och in range(weight.shape[3]):
        for ich in range(weight.shape[2]):
            for ky in range(weight.shape[1]):
                for kx in range(weight.shape[0]):
                    new_weight[kx][ky][ich][och] = ( weight[kx][ky][ich][och] * gamma[och] / np.sqrt(var[och]+1e-3)  )
        new_bias[och] = -mean[och]*gamma1[och]/np.sqrt(var[och]+1e-3)    +     beta[och]
    return new_weight, new_bias

out = ConvBN2ConvBias(weight1[0], beta1, gamma1, mean1, var1)

new_weight1, new_bias1 = out

# redo conv with bias

in_p = np.zeros([1,514,514,3])

for i in range(3):
    for j in range(512):
        for k in range(512):
            in_p[0][j+1][k+1][i] = input_data1[0][j][k][i]
    
    
mmd = np.zeros_like(conv2d_1)

for k in range(16):
    for i in range(17):
        for j in range(17):
            for c in range(3):
                for a in range(3):
                    for b in range(3):
                        mmd[0][i][j][k]  += in_p[0][b+i][a+j][c]*new_weight1[b][a][c][k]
            mmd[0][i][j][k] += new_bias1[k]

m = np.ones([1,17,17,512])*-999
z = np.zeros_like(actf6_c)


#pool_stride1_c[0][0][1][1]
for i in range(512):
    for j in range(16):
        for k in range(16):
            m[0][j][k][i] = actf6_c[0][j][k][i]
for i in range(512):
    for j in range(16):
        for k in range(16):
            z[0][j][k][i] = max(m[0][j][k][i], m[0][j][k+1][i], m[0][j+1][k][i], m[0][j+1][k+1][i])

for i in range(512):
    for j in range(16):
        for k in range(16):
            if(z[0][j][k][i]==pool_stride1_c[0][j][k][i]):
                print("true")
            else:
                print("False")
                print(j,k,i)



#conv with bias

in_p = np.zeros([1,32,32,256])

for i in range(256):
    for j in range(32):
        for k in range(32):
            in_p[0][j][k][i] = conv2d_1[0][j][k][i]
    



    
mmd2 = np.zeros_like(batch_normalization_1)

for k in range(21):
    for i in range(8):
        for j in range(8):
            for c in range(256):
                for a in range(1):
                    for b in range(1):
                        mmd2[0][i][j][k]  += in_p[0][b+i][a+j][c]*getweight[43][0][b][a][c][k]
            mmd2[0][i][j][k] += getweight[43][1][k]



cnt = 0
cnt_n = 0
for k in range(16):
    for i in range(512):
        for j in range(512):
            if((batch_normalization_1[0][i][j][k]-conv2d_14[0][i][j][k])<0.0001):
                cnt+=1
            else:
                cnt_n+=1
