#include <vector>
#include <iostream>
#include <algorithm>
#include "math.h"

struct Bbox{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int class_id;
    Bbox(int x1_, int y1_, int x2_, int y2_, float s, int c_id) :
      x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s),class_id(c_id)  {};
};


// The name of this function is important for Arduino compatibility.

std::vector<Bbox> nms(std::vector<Bbox>& vecBbox, float threshold);

#define INPUT_IMAGE_WIDTH 160
#define INPUT_IMAGE_HEIGHT 160
#define MODEL_IMAGE_WIDTH 160
#define MODEL_IMAGE_HEIGHT 160
#define NUM_CLASS 10
#define NUM_CANDIDATE_BBOX 1575 // 75+300+1200 diff strides
#define CON_THRED  0.5
#define IOU_THRED  0.4
#define DEQNT(a) (float(0.008005863055586815  * (a + 128)))   //0.004304335452616215 * (q + 128)
#define SCORE_THRESHOLD 0.4
#define MAX_QNT_VALUE 1.097605586051941
inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

extern "C" int post_process(int8_t* output, char * result_str);

#define Output_F 0
#define DBG_LOG 1
int post_process(int8_t* output, char * result_str)
{
  static uint8_t null_count = 0;
  const int dimensions = (NUM_CLASS+5);   //our class is 10 , +5 for (xywh, conf)
  const int rows = NUM_CANDIDATE_BBOX;
  //std::vector<int> class_ids;
  //std::vector<float> confidences;
  std::vector<Bbox> boxes;
  #if Output_F
    float* data = (float*)output;
  #else
    int8_t* data = output;
  #endif

  for (int i = 0; i < rows; i++)
  {
    #if Output_F
    float confidence = data[4];
    #else
    float confidence = DEQNT(data[4]);
    #endif
    if (confidence >= CON_THRED)
    {
      #if Output_F
        float* classes_scores = data+5;
        //printf("Data: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\r\n",
        //data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14]);
      #else
        int8_t* classes_scores = data+5;
        //printf("Data: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\r\n",
        //data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14]);
      #endif
      int maxClassId = -1;
      float maxClassScore = 0;
      for (int k = 0; k < NUM_CLASS; k++)
      {
        #if Output_F
          float score = classes_scores[k];
        #else
          float score = DEQNT(classes_scores[k]);
        #endif
        if (score > maxClassScore)
        {
          maxClassId = k;
          maxClassScore = score;
        }
      }

      if ((maxClassScore) > SCORE_THRESHOLD)
      {
        //confidences.push_back(confidence);
        //class_ids.push_back(maxClassId);
        #if Output_F
        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        #else
        float x = DEQNT(data[0]);
        float y = DEQNT(data[1]);
        float w = DEQNT(data[2]);
        float h = DEQNT(data[3]);
        #endif
        int x1 = (int)(clamp((x-w/2)*INPUT_IMAGE_WIDTH, 0, INPUT_IMAGE_WIDTH));
        int y1 =  (int)(clamp((y-h/2)*INPUT_IMAGE_HEIGHT, 0, INPUT_IMAGE_HEIGHT));
        int x2 = (int)(clamp((x+w/2)*INPUT_IMAGE_WIDTH, 0, INPUT_IMAGE_WIDTH));
        int y2 = (int)(clamp((y+h/2)*INPUT_IMAGE_HEIGHT, 0, INPUT_IMAGE_HEIGHT));
        Bbox box(x1, y1, x2, y2, confidence, maxClassId);
        //#if DBG_LOG
        //       sprintf(result_str,"box1(%f, %f, %f, %f) box2(%d, %d, %d, %d) (confidence: %f, class: %d)\r\n", x, y, w, h, x1, y1, x2, y2, confidence, maxClassId);
        //#endif
        boxes.push_back(box);
      }
    }
    data += dimensions;
  }
  uint8_t IsObject = 0;
  int max_class_id = 0;
  float max_confidence = 0;
  float dist_y = 0;
  //start to do nms for each class
  for (int j=0;j < NUM_CLASS; j++)
  {
    //filter the class id
    std::vector<Bbox> class_boxes;
    for (int i=0; i< boxes.size();i++)
    {
      if (boxes[i].class_id == j)
      class_boxes.push_back(boxes[i]);
    }
    if (class_boxes.size() > 0)
    {
      IsObject = 1;
      std::vector<Bbox> result;
      result = nms(class_boxes, IOU_THRED);
      for (int i=0; i< result.size();i++)
      {
        #if DBG_LOG
          sprintf(result_str,"Boxresult[%d] x1 %d, y1 %d, x2 %d, y2 %d, confidence %f, class %d\r\n",
          i, result[i].x1, result[i].y1, result[i].x2, result[i].y2, result[i].score, result[i].class_id);
        #endif
      }
    }
  }
  return 0;
}

float iou(Bbox box1, Bbox box2) {
    float area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);

    int x11 = std::max(box1.x1, box2.x1);
    int y11 = std::max(box1.y1, box2.y1);
    int x22 = std::min(box1.x2, box2.x2);
    int y22 = std::min(box1.y2, box2.y2);
    float intersection = (x22 - x11 + 1) * (y22 - y11 + 1);

    return intersection / (area1 + area2 - intersection);
}

std::vector<Bbox> nms(std::vector<Bbox>& vecBbox, float threshold)
{
  auto cmpScore = [](Bbox box1, Bbox box2) {
    return box1.score < box2.score; // 升序排列, 令score最大的box在vector末端
  };
  std::sort(vecBbox.begin(), vecBbox.end(), cmpScore);
  std::vector<Bbox> pickedBbox;
  //#if DBG_LOG
  //    for (int i=0; i< vecBbox.size();i++)
  //    {
  //   printf("after sort vecBbox[%d]: x1: %d, y1: %d, x2: %d, y2: %d, confidence: %f, class: %d\r\n",
  //       i, vecBbox[i].x1, vecBbox[i].y1, vecBbox[i].x2, vecBbox[i].y2, vecBbox[i].score, vecBbox[i].class_id);
  //    }
  //#endif
  while (vecBbox.size() > 0)
  {
    pickedBbox.emplace_back(vecBbox.back());
    vecBbox.pop_back();
    //#if DBG_LOG
    //        printf("dump vecBBbox size: %d\r\n",vecBbox.size());
    //#endif
    for (size_t i = 0; i < vecBbox.size(); i++)
    {
      float iou_score = iou(pickedBbox.back(), vecBbox[i]);
      //#if DBG_LOG
      //            printf("dump vecBBbox is %d, iou_score: %f\r\n", i, iou_score);
      //#endif
      if (iou_score >= threshold)
      {
        vecBbox.erase(vecBbox.begin() + i);
        i--;
      }
    }
    //#if DBG_LOG
    //        printf("while end and reloop\r\n");
    //#endif
  }
  return pickedBbox;
}