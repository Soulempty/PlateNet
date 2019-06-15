#coding:utf-8
import numpy as np
from detection import Detection
from recognition import Recognition
from PIL import Image,ImageDraw,ImageFont
import cv2
import os 
import sys
import argparse


def main(args):
    img_path = args.img_path
    save_path = args.save_path
    caffe_model = args.caffe_model
    caffe_weight = args.caffe_weight
    ocr_weight = args.ocr_weight
    
    detect = Detection(caffe_model,caffe_weight,device=args.device,conf=0.4)
    recognize = Recognition(ocr_weight)

    imgs = os.listdir(img_path)
    for p in imgs:
        path = os.path.join(img_path,p)
        img = cv2.imread(path)
        imgP = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(imgP)
        h,w = img.shape[:2]       
        if h<800 or w<800:
            r = max(1024//h,1024//w)
            image = cv2.resize(img,(w*r,h*r),interpolation=cv2.INTER_LINEAR)
        else:
            image = img
        top_conf,top_xmin,top_ymin,top_xmax,top_ymax = detect.doDetect(image)
        if top_conf.shape[0]!=0:
            for i in range(top_conf.shape[0]):
                det_conf = round(top_conf[i],2)
                xmin = max(int(round(top_xmin[i] * w)),0)
                ymin = max(int(round(top_ymin[i] * h)),0)
                xmax = min(int(round(top_xmax[i] * w)),w)
                ymax = min(int(round(top_ymax[i] * h)),h)
                crop_img = img[ymin:ymax,xmin:xmax]
                name,acc,ch_conf = recognize.doRecognize(crop_img)
                if len(name)>=9:
                    print("["+p+"]["+str(name)+"][ch:"+str(round(ch_conf,2))+"][rec_conf:"+str(round(acc,2))+"][det_conf:"+str(det_conf)+"]")
                    draw.polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)], outline=(0,255,0))
                    fontText = ImageFont.truetype("./font/simhei.ttf", 30)
                    draw.text((max(xmin-30,5),ymin-30), unicode(name,'utf-8'), (255,0,255), font=fontText)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save = os.path.join(save_path,p)
            imgP.save(save)
            del draw

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='fintune Training')
    parser.add_argument('--img_path',default='./imgs', help='path to the images')
    parser.add_argument('--save_path',default='./results', help='path to the result')
    parser.add_argument('--caffe_model',default='./model/deploy.prototxt', help='path to caffe model')
    parser.add_argument('--caffe_weight',default='./model/plr_iter_40000.caffemodel', help='path to caffe weight')
    parser.add_argument('--ocr_weight',default='./model/weights.50-0.32.h5', help='path to ocr weight')
    parser.add_argument('--device',default='gpu', help='use (gpu/cpu)')
    args = parser.parse_args()
    main(args)
