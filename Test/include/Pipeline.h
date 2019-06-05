

#ifndef PIPLINE_H
#define PIPLINE_H

#include "Plate_Detect.h"
#include "PlateRecognizer.h"
#include <vector>
#include <string>
#include <fstream>
#include "CvxText.h"
#include <iomanip>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>



using namespace std;
using namespace cv;

const vector<string> CH_PLATE_CODE{"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y","Z","港","学","使","警","澳","挂","军","北","南","广","沈","兰","成","济","海","民","航","空"}
;//31+10+24+18                    

class PipelinePR{
  public:  
    PlateRecognizer *plateRecognizer;//无分割识别
    Detector *detector;

    PipelinePR(string detect_prototxt,string detect_caffemodel,
               string recognizer_prototxt,string recognizer_caffemodel,string use_mode
               );//构造函数初始化
    ~PipelinePR();
    int DetectRecognize(Mat img,bool &flag,string name);//图像处理函数
    int VideoRecognize(VideoCapture capture,string out_path,bool save,float scale);//保存到视频文件夹
  private:
    double totalTime;
};

#endif //PIPLINE_H
