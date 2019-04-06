# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:05:14 2019

@author: Ray
"""


import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model_bp import yolo_eval, yolo_body, tiny_yolo_body, tiny_yolo_body_noBN, tiny_yolo_body_noBN_quant_intput
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from keras.models import Model

def ConvBN2ConvBias(weight, beta, gamma, mean, var):
    
    new_weight = np.zeros_like(weight)
    new_bias = np.zeros(weight.shape[3])
    

    for och in range(weight.shape[3]):
        for ich in range(weight.shape[2]):
            for ky in range(weight.shape[1]):
                for kx in range(weight.shape[0]):
                    new_weight[kx][ky][ich][och] = ( weight[kx][ky][ich][och] * gamma[och] / np.sqrt(var[och]+1e-3)  )
        new_bias[och] = -mean[och]*gamma[och]/np.sqrt(var[och]+1e-3)    +     beta[och]
    return new_weight, new_bias


def quant_weight(org_weight, p_bit, n_bit, valid):
    if valid:
        x_ = org_weight * (2**n_bit)
        x_ = np.array(x_, dtype="int32")
        x_ = np.clip(x_, -(2**(p_bit+n_bit)), 2**(p_bit+n_bit)-1)
        x_ = np.array(x_, dtype="float32")
        x_ = x_ / (2**n_bit)
    else:
        x_ = org_weight
    return x_

class YOLO(object):
    _defaults = {
        #"model_path": 'model_data/yolov3-tiny_obj_final_noBN.h5',
        "model_path": 'model_data/yolov3-tiny_obj_final_noBN_quant_bias.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors_shrimp.txt',
        "classes_path": 'model_data/shrimp.names',
        "score" : 0.001,
        "iou" : 0.3,
        "model_image_size" : (512, 512),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
    
    def set_pnbit_valid(self, P_BIT_LIST, N_BIT_LIST, VALID_LIST):
        self.p_bit_list = P_BIT_LIST
        self.n_bit_list = N_BIT_LIST
        self.valid_list = VALID_LIST
        
    def set_model_parameter(self, Pbit, Nbit, VALID ):
        self.p_bit = Pbit
        self.n_bit = Nbit
        self.valid = VALID
        self.boxes, self.scores, self.classes = self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body_noBN_quant_intput(Input(shape=(None,None,3)), num_anchors//2, num_classes, self.p_bit, self.n_bit ) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        self.yolo_model.summary()
        
        
        
        
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        
        
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        
        
        
        getconfig = []
        getweight = []
        for layer in self.yolo_model.layers:
            getconfig.append(layer.get_config())
            getweight.append(layer.get_weights())
            
        #intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_23").output)
        #conv2d_23 = intermediate_layer_model.predict(image_data)
        
        #intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_26").output)
        #conv2d_26 = intermediate_layer_model.predict(image_data)
        
        #intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("input_1").output)
        #input_data1 = intermediate_layer_model.predict(image_data)
        
        
        end = timer()
        
        print(end - start)
        return image, getconfig, getweight, 0, 0, 0, 0,image_data
    
    def detect_image_box(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #thickness = (image.size[0] + image.size[1]) // 300
        
        result_list = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #draw = ImageDraw.Draw(image)
            #label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            result_list.append([predicted_class, score, left, top, right, bottom])
            '''
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            '''
            #del draw
        '''
        for layer in self.yolo_model.layers:
            getconfig=layer.get_config()
            getweight=layer.get_weights()
            print (getconfig)
            print (getweight)
        '''    
        
        #intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_1").output)
        #conv2d_1 = intermediate_layer_model.predict(image_data)
        '''
        print('1 : ',intermediate_output)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("lambda_1").output)
        intermediate_output = intermediate_layer_model.predict(image_data)
        print('2 : ',intermediate_output)
        #leaky_re_lu_5
        #lambda_1
        '''
        
        
        end = timer()
        print(end - start)
        return image, result_list, 0
    
    def get_layers_feature(self, image):
        start = timer()
        
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        result_list = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            result_list.append([predicted_class, score, left, top, right, bottom])
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        
        
        
        getconfig = []
        getweight = []
        for layer in self.yolo_model.layers:
            getconfig.append(layer.get_config())
            getweight.append(layer.get_weights())
        
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_1").output)
        conv2d_1 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_1").output)
        leaky_re_lu_1 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_2").output)
        conv2d_2 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_2").output)
        leaky_re_lu_2 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_3").output)
        conv2d_3 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_3").output)
        leaky_re_lu_3 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_4").output)
        conv2d_4 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_4").output)
        leaky_re_lu_4 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_5").output)
        conv2d_5 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_5").output)
        leaky_re_lu_5 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_6").output)
        conv2d_6 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_6").output)
        leaky_re_lu_6 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_7").output)
        conv2d_7 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_7").output)
        leaky_re_lu_7 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_8").output)
        conv2d_8 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_8").output)
        leaky_re_lu_8 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_9").output)
        conv2d_9 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_9").output)
        leaky_re_lu_9 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_10").output)
        conv2d_10 = intermediate_layer_model.predict(image_data)
        
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_11").output)
        conv2d_11 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_10").output)
        leaky_re_lu_10 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_12").output)
        conv2d_12 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("leaky_re_lu_11").output)
        leaky_re_lu_11 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_13").output)
        conv2d_13 = intermediate_layer_model.predict(image_data)
        
        

        
        layers_feature_list = [ image_data, conv2d_1, leaky_re_lu_1, conv2d_2, leaky_re_lu_2, conv2d_3, leaky_re_lu_3, conv2d_4, leaky_re_lu_4,
                               conv2d_5, leaky_re_lu_5, conv2d_6, leaky_re_lu_6, conv2d_7, leaky_re_lu_7, conv2d_8, leaky_re_lu_8, conv2d_9, leaky_re_lu_9,
                               conv2d_10, conv2d_11, leaky_re_lu_10, conv2d_12, leaky_re_lu_11, conv2d_13 ]
        
        
        
        end = timer()
        
        print(end - start)
        return image, result_list, layers_feature_list
    
    # The quant bias's p_bit/n_bit is as same as input feature.   'cause of hardware design
    def save_quant_bias(self, image): 
        start = timer()
        #####test
        
        getconfig = []
        getweight = []
        for layer in self.yolo_model.layers:
            getconfig.append(layer.get_config())
            getweight.append(layer.get_weights())
            #print (getconfig)
            #print (getweight)
        
        
        convbias_1 = [ getweight[1][0], quant_weight(getweight[1][1], 7, 8, 1) ]  #conv2d_1
        convbias_2 = [ getweight[6][0], quant_weight(getweight[6][1], 4, 11, 1) ]  #conv2d_2   
        convbias_3 = [ getweight[11][0], quant_weight(getweight[11][1], 4, 11, 1) ]  #conv2d_3
        convbias_4 = [ getweight[16][0], quant_weight(getweight[16][1], 5, 10, 1) ]  #conv2d_4
        convbias_5 = [ getweight[21][0], quant_weight(getweight[21][1], 3, 12, 1) ]  #conv2d_5
        convbias_6 = [ getweight[26][0], quant_weight(getweight[26][1], 4, 11, 1) ]  #conv2d_6
        convbias_7 = [ getweight[31][0], quant_weight(getweight[31][1], 4, 11, 1) ]  #conv2d_7
        convbias_8 = [ getweight[35][0], quant_weight(getweight[35][1], 4, 11, 1) ]  #conv2d_8
        convbias_9 = [ getweight[45][0], quant_weight(getweight[45][1], 5, 10, 1) ]  #conv2d_9
        convbias_10 = [ getweight[53][0], quant_weight(getweight[53][1], 5, 10, 1) ]  #conv2d_10
        convbias_11 = [ getweight[39][0], quant_weight(getweight[39][1], 4, 11, 1) ]  #conv2d_11
        convbias_12 = [ getweight[46][0], quant_weight(getweight[46][1], 4, 11, 1) ]  #conv2d_12
        convbias_13 = [ getweight[54][0], quant_weight(getweight[54][1], 4, 11, 1) ]  #conv2d_13
       
       
        print('success1')
        self.yolo_model.get_layer('conv2d_1').set_weights(convbias_1)
        
        print('success2')
        self.yolo_model.get_layer('conv2d_2').set_weights(convbias_2)
        print('success3')
        self.yolo_model.get_layer('conv2d_3').set_weights(convbias_3)
        print('success4')
        self.yolo_model.get_layer('conv2d_4').set_weights(convbias_4)
        print('success5')
        self.yolo_model.get_layer('conv2d_5').set_weights(convbias_5)
        
        print('success6')
        self.yolo_model.get_layer('conv2d_6').set_weights(convbias_6)
        print('success7')
        self.yolo_model.get_layer('conv2d_7').set_weights(convbias_7)
        print('success8')
        self.yolo_model.get_layer('conv2d_8').set_weights(convbias_8)
        print('success9')
        self.yolo_model.get_layer('conv2d_9').set_weights(convbias_9)
        print('success10')
        self.yolo_model.get_layer('conv2d_10').set_weights(convbias_10)
        print('success11')
        self.yolo_model.get_layer('conv2d_11').set_weights(convbias_11)
        print('success12')
        self.yolo_model.get_layer('conv2d_12').set_weights(convbias_12)
        print('success13')
        self.yolo_model.get_layer('conv2d_13').set_weights(convbias_13)
        print('success all')
        
        
        ####test
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        
        
        
        # save 
        #self.yolo_model.save_weights('model_data/yolov3-tiny_obj_final_noBN_quant_bias.h5')
        
        getconfig2 = []
        getweight2 = []
        for layer in self.yolo_model.layers:
            getconfig2.append(layer.get_config())
            getweight2.append(layer.get_weights())
            
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_10").output)
        conv2d_23 = intermediate_layer_model.predict(image_data)
        
        intermediate_layer_model = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer("conv2d_13").output)
        conv2d_26 = intermediate_layer_model.predict(image_data)
        


        end = timer()
        
        print(end - start)
        return image, getconfig, getweight, getconfig2, getweight2, conv2d_23, conv2d_26,image_data
    def get_weights(self):
        getconfig = []
        getweight = []
        for layer in self.yolo_model.layers:
            getconfig.append(layer.get_config())
            getweight.append(layer.get_weights())
        return getconfig, getweight
    def search_layers_weight(self, image, getweight):
        start = timer()
        #####test
        '''
        getconfig = []
        getweight = []
        for layer in self.yolo_model.layers:
            getconfig.append(layer.get_config())
            getweight.append(layer.get_weights())
            #print (getconfig)
            #print (getweight)
        '''
        

        
        
        q_convwegt_1 = [ quant_weight(getweight[1][0], self.p_bit_list[0], self.n_bit_list[0], self.valid_list[0]), getweight[1][1]]        # conv2d_1
        
        q_convwegt_2 = [ quant_weight(getweight[6][0], self.p_bit_list[1], self.n_bit_list[1], self.valid_list[1]), getweight[6][1]]        # conv2d_2
        
        q_convwegt_3 = [ quant_weight(getweight[11][0], self.p_bit_list[2], self.n_bit_list[2], self.valid_list[2]), getweight[11][1]]      # conv2d_3
        
        q_convwegt_4 = [ quant_weight(getweight[16][0], self.p_bit_list[3], self.n_bit_list[3], self.valid_list[3]), getweight[16][1]]      # conv2d_4
        
        q_convwegt_5 = [ quant_weight(getweight[21][0], self.p_bit_list[4], self.n_bit_list[4], self.valid_list[4]), getweight[21][1]]      # conv2d_5
        
        q_convwegt_6 = [ quant_weight(getweight[26][0], self.p_bit_list[5], self.n_bit_list[5], self.valid_list[5]), getweight[26][1]]      # conv2d_6
        
        q_convwegt_7 = [ quant_weight(getweight[31][0], self.p_bit_list[6], self.n_bit_list[6], self.valid_list[6]), getweight[31][1]]      # conv2d_7
        
        q_convwegt_8 = [ quant_weight(getweight[35][0], self.p_bit_list[7], self.n_bit_list[7], self.valid_list[7]), getweight[35][1]]      # conv2d_8
        
        q_convwegt_9 = [ quant_weight(getweight[45][0], self.p_bit_list[8], self.n_bit_list[8], self.valid_list[8]), getweight[45][1]]      # conv2d_9
        
        q_convwegt_10 = [ quant_weight(getweight[53][0], self.p_bit_list[9], self.n_bit_list[9], self.valid_list[9]), getweight[53][1]]     # conv2d_10
        
        q_convwegt_11 = [ quant_weight(getweight[39][0], self.p_bit_list[10], self.n_bit_list[10], self.valid_list[10]), getweight[39][1]]  # conv2d_11
        
        q_convwegt_12 = [ quant_weight(getweight[46][0], self.p_bit_list[11], self.n_bit_list[11], self.valid_list[11]), getweight[46][1]]  # conv2d_12
        
        q_convwegt_13 = [ quant_weight(getweight[54][0], self.p_bit_list[12], self.n_bit_list[12], self.valid_list[12]), getweight[54][1]]  # conv2d_13
        
        
        #print(self.p_bit_list)
        #print(self.n_bit_list)
        #print(self.valid_list)
        #print('success1')
        self.yolo_model.get_layer('conv2d_1').set_weights(q_convwegt_1)
        
        #print('success2')
        self.yolo_model.get_layer('conv2d_2').set_weights(q_convwegt_2)
        #print('success3')
        self.yolo_model.get_layer('conv2d_3').set_weights(q_convwegt_3)
        #print('success4')
        self.yolo_model.get_layer('conv2d_4').set_weights(q_convwegt_4)
        #print('success5')
        self.yolo_model.get_layer('conv2d_5').set_weights(q_convwegt_5)
        
        #print('success6')
        self.yolo_model.get_layer('conv2d_6').set_weights(q_convwegt_6)
        #print('success7')
        self.yolo_model.get_layer('conv2d_7').set_weights(q_convwegt_7)
        #print('success8')
        self.yolo_model.get_layer('conv2d_8').set_weights(q_convwegt_8)
        #print('success9')
        self.yolo_model.get_layer('conv2d_9').set_weights(q_convwegt_9)
        #print('success10')
        self.yolo_model.get_layer('conv2d_10').set_weights(q_convwegt_10)
        #print('success11')
        self.yolo_model.get_layer('conv2d_11').set_weights(q_convwegt_11)
        #print('success12')
        self.yolo_model.get_layer('conv2d_12').set_weights(q_convwegt_12)
        #print('success13')
        self.yolo_model.get_layer('conv2d_13').set_weights(q_convwegt_13)
        #print('success all')
        
        
        ####test
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        result_list = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            result_list.append([predicted_class, score, left, top, right, bottom])
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        
        
        end = timer()
        
        getconfig2 = []
        getweight2 = []
        for layer in self.yolo_model.layers:
            getconfig2.append(layer.get_config())
            getweight2.append(layer.get_weights())
            #print (getconfig)
            #print (getweight)
            
        
        #print(end - start)
        return image, result_list, getconfig2, [ getweight, getweight2 ]
    
    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

