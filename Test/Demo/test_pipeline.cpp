#include "Pipeline.h"
#include <vector>
#include <string>
#include <fstream>
using namespace std;
using namespace cv;

int main(int argc,char** argv){
  const char *path;
  float scale=0.5;
  if(argc>=2)
    path=argv[1];

  string detect_prototxt   = "./models/deploy.prototxt";
  string detect_caffemodel = "./models/plr_iter_40000.caffemodel"; //for visualization
  string recognizer_prototxt = "./models/SegmentationFree.prototxt";
  string recognizer_caffemodel = "./models/SegmentationFree.caffemodel";
  string use_mode;
  bool flag=false;
  if(argc==8)
    {detect_prototxt = argv[4];
     detect_caffemodel = argv[5];
     recognizer_prototxt = argv[6];
     recognizer_caffemodel = argv[7];
    }
  if(argc>=3)
    scale=atof(argv[2]);
  if(argc>=4)
    use_mode=argv[3];
  else
    use_mode="cpu";
  PipelinePR pipeline(detect_prototxt, detect_caffemodel,
                      recognizer_prototxt, recognizer_caffemodel, use_mode);
  
  struct stat s_buf;
  stat(path,&s_buf);
  string f(path);
  if(S_ISDIR(s_buf.st_mode)){
    struct dirent *ptr;
    DIR *dir;
    dir=opendir(path);
    while((ptr=readdir(dir))!=NULL)
    {
      if(ptr->d_name[0] == '.')
        continue;
      string img=f+"/"+ptr->d_name;
      int num;
      if(img.find(".jpg")<img.length() || img.find(".png")<img.length() || img.find(".jpeg")<img.length())
        {Mat image = imread(img);
         num = pipeline.DetectRecognize(image,flag,ptr->d_name);
        }
      if(flag)
        break;
    }
  }
  else if(S_ISREG(s_buf.st_mode)&&(f.find(".jpg")<f.length() || f.find(".png")<f.length() || f.find(".jpeg")<f.length()))
    {
     Mat img = imread(f);
     int pos=f.find_last_of('/');
     string name(f.substr(pos+1));
     int num = pipeline.DetectRecognize(img,flag,name);
     string file ;
     cout<<"输入图片路径：";
     while(cin>>file)//获取图片路径
      {if(file.find(".jpg")<file.length() || file.find(".png")<file.length() || file.find(".jpeg")<file.length())
        {Mat img =  imread(file);//opencv读取图片
         int pos=file.find_last_of('/');
         string name(file.substr(pos+1));
         int num = pipeline.DetectRecognize(img,flag,name);
      }
      
      if(flag)
        break;
      }    
      cout<<"输入图片路径：";
   }
  else if(f.find(".mp4")<f.length() || f.find(".avi")<f.length() || f.find(".mkv")<f.length())
  {
    VideoCapture capture(f);
    string h=f.substr(f.find_last_of('/')+1,f.size()-f.find_last_of('/')-1);
    string out_file="./"+h.replace(h.find_last_of('.'),4,".avi");
    pipeline.VideoRecognize(capture,out_file,true,scale);//VideoCapture capture,string out_path,bool save=true,float scale=1.0
  }
  return 0;
}


