#coding:utf-8
import numpy as np
import cv2,os,sys

caffe_root='../PlateNet/python'
sys.path.insert(0, caffe_root)
import caffe

class Detection():
    def __init__(self,caffe_model,caffe_weight,device='gpu',conf=0.5):
        self.net = caffe.Net(caffe_model, caffe_weight, caffe.TEST)
        self.conf = conf
        if device=='gpu':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu() 
    def doDetect(self,image):
        
        self.net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([109, 111, 105])) # mean pixel  
        transformed_image = transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        detections = self.net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.conf]
        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        return top_conf,top_xmin,top_ymin,top_xmax,top_ymax


