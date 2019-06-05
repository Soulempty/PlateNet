

#include "Plate_Detect.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

//构造函数
Detector::Detector(const string& model_file,
                   const string& trained_file,string device_mode="cpu") {

  if(device_mode=="gpu")
  {Caffe::set_mode(Caffe::GPU);}
  else
  {Caffe::set_mode(Caffe::CPU);}

  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ =  Size(input_layer->width(), input_layer->height());
}

//预测功能函数
vector<vector<float> > Detector::doDetect(const  Mat& img) {
  vector< Mat> input_channels;
  vector<Blob<float>*> input_vec;

  Init_img(img, &input_channels);

  net_->Forward(input_vec);//need parameters
  /* Copy the output layer to a  vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
 
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}
void Detector::get_plates(vector<vector<float> > detections,vector<plateInfo> &plates,float confidence_threshold=0.5){
  int width = temp_img.cols;
  int height = temp_img.rows;
  Mat cropped;
  plateInfo plateinfo;
  for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        float score = d[2];
        int label = static_cast<int>(d[1]);
        if (score >= confidence_threshold&&label==1) {
          
          int det_xmin = static_cast<int>(d[3] * width);
          int det_ymin = static_cast<int>(d[4] * height);
          int det_xmax = static_cast<int>(d[5] * width);
          int det_ymax = static_cast<int>(d[6] * height);
          det_xmin = max(det_xmin-6,0);
          det_ymin = max(det_ymin-2,0);
          det_xmax = min(det_xmax+6,width);
          det_ymax = min(det_ymax+2,height);
          int det_width,det_height;
          det_width = det_xmax-det_xmin;
          det_height = det_ymax-det_ymin;
          float ratio = min((float)det_width/width,(float)det_height/height);
          if (ratio<0.30)
          {
            Rect rect(det_xmin,det_ymin,det_width,det_height);
            Mat temp(rect.size(),temp_img.type());
            temp = temp_img(rect);
            temp.copyTo(cropped);
            plateinfo.score = score;
            plateinfo.rect = rect;
            plateinfo.cropped = cropped;
            plates.push_back(plateinfo);
          }
        }
      }
}
//可以放在构造函数中直接初始化
void Detector::Init_img(const  Mat& img,vector< Mat>* input_channels) {
  //Convert the input image to the input image format of the network.

  Mat img_resized;
  input_geometry_ = Size(img.cols,img.rows);
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  float* input_data = input_layer->mutable_cpu_data();//cpu数据地址
  int width = input_layer->width();
  int height = input_layer->height();

  for (int i = 0; i < input_layer->channels(); ++i) {
    Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
  temp_img = img;//保存图像为全局
  Mat img_float;
  img.convertTo(img_float, CV_32FC3);
  Mat img_norm;
  subtract(img_float,  Scalar(109, 111, 105), img_norm);//102, 107, 104
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the  Mat
   * objects in input_channels. */
  split(img_norm, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void Detector::Visualization(vector<vector<float> > detections,float confidence_threshold,bool &flag) {

  namedWindow("PLD",0);
  int width = temp_img.cols;
  int height = temp_img.rows;
  for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
          int label = static_cast<int>(d[1]);
          string text;// = to_string(score);
          ostringstream os;
          os<<score;
          text.append(os.str()); 
          int det_xmin = static_cast<int>(d[3] * width) - 2 > width ? static_cast<int>(d[3] * width) -2 : static_cast<int>(d[3] * width);
          int det_ymin = static_cast<int>(d[4] * height) - 2 > height ? static_cast<int>(d[4] * height) -2 : static_cast<int>(d[4] * height);
          int det_xmax = static_cast<int>(d[5] * width) + 2 < width ? static_cast<int>(d[5] * width) +2 : static_cast<int>(d[5] * width);
          int det_ymax = static_cast<int>(d[6] * height) +2 < height ? static_cast<int>(d[6] * height) +2 : static_cast<int>(d[6] * height);
          rectangle(temp_img,Point(det_xmin,det_ymin),cv::Point(det_xmax,det_ymax),cv::Scalar(255,255,0),2);
          putText(temp_img, text, Size(det_xmin-4,det_ymin-4), FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 255), 2, 1, 0);
        }
      }
  imshow("PLD",temp_img);
  while(true)
    {char key = (char)waitKey(1);
     if(key == 'd')
       {
        break;}
     if(key=='q')
     {flag=true;
      destroyAllWindows();
      break;
     }
   }
}

