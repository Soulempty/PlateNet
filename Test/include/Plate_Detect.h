
#ifndef PLATE_DETECT_HPP
#define PLATE_DETECT_HPP
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory> 
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  
using namespace std;
using namespace cv;

typedef struct plateInfo{  
   float score;  
   Mat cropped;  
   Rect rect;   
} plateInfo;  

class Detector {
 public:
  Detector(const string& model_file,
             const string& trained_file,string device_mode);//初始化网络设置


  vector<vector<float> > doDetect(const  Mat& img);//预测主函数
  void Visualization( vector<vector<float> > detections,float confidence_threshold,bool &flag);//opencv处理图像显示

  void get_plates(vector<vector<float> > detections,vector<plateInfo> &plates,float confidence_threshold);

 private:
  void Init_img(const  Mat& img,vector< Mat>* input_channels);//图像初始化辅助函数

 private:
  boost::shared_ptr<Net<float> > net_;
  Size input_geometry_;
  int num_channels_;
  Blob<float>* input_layer;
  Mat temp_img; 
};

#endif
