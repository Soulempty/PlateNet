
#include "Pipeline.h"
#include  <stdio.h> 

static int ToWchar(char* &src, wchar_t* &dest, const char *locale = "zh_CN.utf8")
{
    if (src == NULL) {
        dest = NULL;
        return 0;
    }
    setlocale(LC_CTYPE, locale);
    int w_size = mbstowcs(NULL, src, 0) + 1;
    if (w_size == 0) {
        dest = NULL;
        return -1;
    }
    dest = new wchar_t[w_size];
    if (!dest) {
        return -1;
    }
    int ret = mbstowcs(dest, src, strlen(src)+1);
    if (ret <= 0) {
        return -1;
    }
    return 0;
}

PipelinePR::PipelinePR(string detect_prototxt,string detect_caffemodel,
                       string recognizer_prototxt,string recognizer_caffemodel,string use_mode)
    {
        detector = new Detector(detect_prototxt,detect_caffemodel,use_mode);
        plateRecognizer = new PlateRecognizer(recognizer_prototxt,recognizer_caffemodel);   
    }
PipelinePR::~PipelinePR() 
  {
    delete plateRecognizer;
    delete detector;
  }
int PipelinePR::DetectRecognize(Mat img,bool &flag,string name)
  {
        
    vector<plateInfo> plates;
    Mat img_scale;
    float s = 1.0;
    //namedWindow("PlateRecognize",0);
    CvxText text("./src/font/simhei.ttf"); //指定字体
    //计时
    Scalar size1{ 40, 0.5, 0.1, 0 }; // (字体大小, 无效的, 字符间距, 无效的 }

    text.setFont(nullptr, &size1, nullptr, 0);
    struct timeval time;
    gettimeofday(&time, NULL); // Start Time
    totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
    if (img.cols<800 || img.rows<800)
    { 
      s = max(896.0/img.cols,896.0/img.rows);
      resize(img,img_scale,Size((int)(img.cols*s),(int)(img.rows*s)));
    }
    else
      img_scale = img;
    
    vector<vector<float> > detections = detector->doDetect(img_scale);
    detector->get_plates(detections,plates,0.4);   
    int num = plates.size();
    for (plateInfo plateinfo:plates)
    {
      Mat cropped = plateinfo.cropped;
      Rect rect = plateinfo.rect;
      float score = plateinfo.score;
      pair<string,float> res = plateRecognizer->doRecognize(cropped,CH_PLATE_CODE);
      string nam = res.first;
      float rec_score = res.second;  
      if(rec_score<0.75)  
        nam = "Vague";    
      cout <<"["<<name <<"]["<<nam<< "][识别度" << rec_score << "][检测度"<<score <<"]"<<endl;  
      rectangle(img,Point(rect.x/s,rect.y/s),Point((rect.x+rect.width)/s,(rect.y+rect.height)/s),cv::Scalar(0,255,0),2);
      char* str = (char *)nam.c_str();////str.c_str()或者str.data()
      wchar_t *w_str;
      ToWchar(str,w_str);
      text.putText(img, w_str, Point(rect.x/s-30,rect.y/s-20), Scalar(255,0,255));
    }
    gettimeofday(&time, NULL);  //END-TIME
    totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
    cout << "Processing time:" << totalTime << " ms" <<  endl;
    string folder = "./result/";
    if (0 != access(folder.c_str(), 0))
        mkdir(folder.c_str(),484);
    imwrite(folder+name,img);
    return num;
}

int PipelinePR::VideoRecognize(VideoCapture capture,string out_path,bool save=true,float scale=1.0)
  {
    
    int width = capture.get(3);
    int height = capture.get(4);
    if (scale != 1.0)
    {
     width*=scale;
     height*=scale;
    }
    CvxText text("./src/font/simhei.ttf"); //指定字体
    Scalar size1{ 60, 0.5, 0.1, 0 }; // (字体大小, 无效的, 字符间距, 无效的 }

    text.setFont(nullptr, &size1, nullptr, 0);
    VideoWriter writer(out_path, CV_FOURCC('D', 'I', 'V', 'X'), 20.0, Size(width, height));
    cout<<"width:"<<width<<"height:"<<height<<endl;
    //namedWindow("PlateRecognize");
    if(!capture.isOpened())
      return -1; 
    while(true)
    {
      Mat frame,img_roi;
      vector<plateInfo> plates;
      capture>>frame;
      if (frame.empty())
       {
	 printf("播放结束\n");
	 break;
        }
      if (scale != 1.0)
        resize(frame, img_roi, Size(width,height));
      else
        img_roi = frame;
      struct timeval time;
      gettimeofday(&time, NULL); // Start Time
      totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
      vector<vector<float> > detecttions = detector->doDetect(img_roi);
      detector->get_plates(detecttions,plates,0.7);
      int num = plates.size();
      
      for (plateInfo plateinfo:plates)
      {
        Mat cropped = plateinfo.cropped;
        Rect rect = plateinfo.rect;
        float score = plateinfo.score;
        pair<string,float> res = plateRecognizer->doRecognize(cropped,CH_PLATE_CODE);
        string name = res.first;
        float rec_score = res.second;
        
        cout <<name << " 识别度：" << rec_score << " 检测度："<<score <<endl;  
        rectangle(img_roi,Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height),Scalar(0,255,0),2);
        char* str = (char *)name.c_str();////str.c_str()或者str.data()
        wchar_t *w_str;
        ToWchar(str,w_str);
        text.putText(img_roi, w_str, Point(rect.x-30,rect.y-20), Scalar(255,0,255));
        
      }
      if (save)
          writer<<img_roi;
      //imshow("PlateRecognize",img_roi);
      gettimeofday(&time, NULL);  //END-TIME
      totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
      cout << "Processing time:" << totalTime << " ms" <<  endl;
      
      //char f = char(waitKey(1));
      //if (f =='q')
        //{break;
        // destroyAllWindows();
        //}
    }  
    //destroyAllWindows();
    return 0;  
  }
  
