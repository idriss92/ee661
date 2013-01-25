#ifndef __HAARBY_h__
#define __HAARBY_h__

#include <iostream>
#include <string>
#include "cv.h"

#ifdef _WIN32
#include <windows.h>
#include <stack>
#include <iostream>
#endif

using namespace std;

namespace CVisby{
  const int HaarW = 40;
  const int HaarH = 20;
  const int HaarSize = 166000;
  const int HaarPts = 12;
  const double AlphaINF = 1000;

  struct DecisionStump{
    int p;
    double theta;
  };

  class IndDouble{
  public:
    IndDouble(){ind=0;value=0;};
    int ind;
    double value;
  };
  bool ListFiles(string path, string mask, vector<string>& files);
  bool IndDoublePredicate(const IndDouble& a, const IndDouble& b);

  class Haarby{
  public:
    IplImage* pImage;
    int beginIndex;
    int numBatch;
    double* features;
    double* integralImage;
    double* integralS;
    int* featureIndex;
    double* thisBatchFeatures;
    string outDn;
    string outFn;
    bool storeBinary;
    int numFiles;
    int iterFiles;
    Haarby():features(NULL),integralImage(NULL),integralS(NULL),featureIndex(NULL),thisBatchFeatures(NULL){};
    void init(int batch,string dn,bool binaryMode = false, int numF = 0);
    void setBatchIndex(int bI);
    void calcFeatures(IplImage* image);
    void getIntegralImage();
    void getFeatures(IplImage* image, int numFeatures, int* indFeatures, double* outFeatures);
    void destroy();
    void saveBinaryFile();
    void readBinaryFile(string fn, double* bMem);

  private:


  };

}

#endif