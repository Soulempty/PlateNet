

#ifndef SWIFTPR_FASTDESKEW_H
#define SWIFTPR_FASTDESKEW_H

#include <math.h>
#include <opencv2/opencv.hpp>

cv::Mat fastdeskew(cv::Mat skewImage,int blockSize);
#endif //SWIFTPR_FASTDESKEW_H
