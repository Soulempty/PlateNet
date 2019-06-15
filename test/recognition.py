#coding=utf-8
from keras import backend as K
from keras.models import *
from keras.layers import *
import numpy as np
import random
import string
import os
import cv2

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z","港","学","使","警","澳","挂","军","北","南","广","沈","兰","成","济","海","民","航","空"
             ]


class Recognition():
    def __init__(self,ocr_weight):
        self.ocr_model = construct_model(ocr_weight)
        self.cls = len(chars)
    def fastdecode(self,y_pred):
        results = ""
        conf = []
        ch_conf = 0.0
        table_pred = y_pred.reshape(-1, self.cls+1)
        res = table_pred.argmax(axis=1)
        for i,one in enumerate(res):
            if one<self.cls and (i==0 or (one!=res[i-1])):
                results+= chars[one]
                conf.append(table_pred[i][one])
        acc = sum(conf)/len(conf)
        ch_conf = conf[0]
        return results,acc,ch_conf

    def doRecognize(self,crop_img):
        img = cv2.resize(crop_img,( 160,40),interpolation=cv2.INTER_LINEAR)
        img = img.transpose(1, 0, 2)
        res = self.ocr_model.predict(np.array([img]))
        res = res[:,:,:]
        result = self.fastdecode(res)
        return result



def construct_model(model_path):
    input_tensor = Input((None, 40, 3))
    x = input_tensor
    base_conv = 32

    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    conv2d_1 = Conv2D(256, (1, 5))(x)
    bn1 = BatchNormalization()(conv2d_1)
    bn1 = Activation('relu')(bn1)
    
    conv2d_2 = Conv2D(256, (7, 1),padding='same')(bn1)
    bn2 = BatchNormalization()(conv2d_2)
    bn2 = Activation('relu')(bn2)

    conv2d_3 = Conv2D(256, (5, 1),padding='same')(bn1)
    bn3 = BatchNormalization()(conv2d_3)
    bn3 = Activation('relu')(bn3)

    conv2d_4 = Conv2D(256, (3, 1),padding='same')(bn1)
    bn4 = BatchNormalization()(conv2d_4)
    bn4 = Activation('relu')(bn4)

    conv2d_5 = Conv2D(256, (1, 1),padding='same')(bn1)
    bn5 = BatchNormalization()(conv2d_5)
    bn5 = Activation('relu')(bn5)

    concat = Concatenate(axis=-1)([bn1,bn2,bn3,bn4,bn5])
    conv2d_f = Conv2D(1024, (1, 1))(concat)
    bn_f = BatchNormalization()(conv2d_f)
    bn_f = Activation('relu')(bn_f)

    cls = Conv2D(84, (1, 1))(bn_f)
    cls = Activation('softmax')(cls)

    base_model = Model(inputs=input_tensor, outputs=cls)
    base_model.load_weights(model_path)
    return base_model
