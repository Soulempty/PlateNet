
#ifndef SEGMENTATIONFREERECOGNIZER_H
#define SEGMENTATIONFREERECOGNIZER_H

#include "opencv2/dnn.hpp"
#include <opencv2/opencv.hpp>
#include "FastDeskew.h"

using namespace cv;
using namespace std;
class PlateRecognizer{
  public:
    
    PlateRecognizer(string prototxt,string caffemodel);
    pair<string,float> doRecognize(Mat plate,vector<string> mapping_table);

  private:
    dnn::Net net;

    };
#endif //SEGMENTATIONFREERECOGNIZER_H
