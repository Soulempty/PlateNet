//

//
#include "PlateRecognizer.h"
#include <iostream>
using namespace std;

    
PlateRecognizer::PlateRecognizer(string prototxt, string caffemodel) 
  {
    net = dnn::readNetFromCaffe(prototxt, caffemodel);
  }

inline int judgeCharRange(int id)
  {return id<31 || id>64;//确定第一个字符为汉字
  }
pair<string,float> decodeResults(Mat code_table,vector<string> mapping_table,float thres)
  {
    MatSize mtsize = code_table.size;//1x84x20x1
    int sequencelength = mtsize[2];
    int labellength = mtsize[1];
    transpose(code_table.reshape(1,1).reshape(1,labellength),code_table);//(1,1680),(84,20),(20,84)
    string name = "";
    vector<int> seq(sequencelength);//20
    vector<pair<int,float>> seq_decode_res;//保存结果
    for(int i = 0 ; i < sequencelength;  i++) {
      float *fstart = ((float *) (code_table.data) + i * labellength );
      int id = max_element(fstart,fstart+labellength) - fstart;//确定20个序列的类别标签，即概率最大值
      seq[i] =id;//获得20个字符的标签
      }
    float sum_confidence = 0;
    int plate_lenghth  = 0 ;
    for(int i = 0 ; i< sequencelength ; i++)
      {
        if(seq[i]!=labellength-1 && (i==0 || seq[i]!=seq[i-1]))//labellength-1 表示特殊字符
          {
             float *fstart = ((float *) (code_table.data) + i * labellength );
             float confidence = *(fstart+seq[i]);
             pair<int,float> pair_(seq[i],confidence);
             seq_decode_res.push_back(pair_);
          }
      }
    int  i = 0;
    if (seq_decode_res.size()>1 && judgeCharRange(seq_decode_res[0].first) && judgeCharRange(seq_decode_res[1].first))
      {
        i=2;//前两个字符决定汉字
        int c = seq_decode_res[0].second<seq_decode_res[1].second;
        name+=mapping_table[seq_decode_res[c].first];
        sum_confidence+=seq_decode_res[c].second;
        plate_lenghth++;
      }

    for(; i < seq_decode_res.size();i++)
      {
        name+=mapping_table[seq_decode_res[i].first];
        sum_confidence +=seq_decode_res[i].second;
        plate_lenghth++;
      }
     pair<string,float> res;
     res.second = sum_confidence/plate_lenghth;
     res.first = name;
     return res;
   }
string decodeResults(Mat code_table,vector<string> mapping_table)
  {
    MatSize mtsize = code_table.size;
    int sequencelength = mtsize[2];
    int labellength = mtsize[1];
    transpose(code_table.reshape(1,1).reshape(1,labellength),code_table);
    string name = "";
    vector<int> seq(sequencelength);
    for(int i = 0 ; i < sequencelength;  i++) {
      float *fstart = ((float *) (code_table.data) + i * labellength );
      int id = max_element(fstart,fstart+labellength) - fstart;
      seq[i] =id;
      }
    for(int i = 0 ; i< sequencelength ; i++)
      {
       if(seq[i]!=labellength-1 && (i==0 || seq[i]!=seq[i-1]))
         name+=mapping_table[seq[i]];
      }
    return name;
  }
pair<string,float> PlateRecognizer::doRecognize(Mat Image,vector<string> mapping_table) 
  {
    Mat descrew = fastdeskew(Image, 5);
    transpose(descrew,descrew);
    Mat inputBlob = dnn::blobFromImage(descrew, 1 / 255.0, Size(40,160));
    net.setInput(inputBlob, "data");
    Mat char_prob_mat = net.forward();
    return decodeResults(char_prob_mat,mapping_table,0.00);
  }
