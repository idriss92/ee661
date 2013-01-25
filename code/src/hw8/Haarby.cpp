#include "Haarby.h"
#include "cv.h"
#include "cxcore.h"

#include <fstream>

using namespace std;

namespace CVisby{

#ifdef _WIN32
  bool ListFiles(string path, string mask, vector<string>& files) {
    HANDLE hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATA ffd;
    string spec;
    stack<string> directories;

    directories.push(path);
    files.clear();

    while (!directories.empty()) {
      path = directories.top();
      spec = path + "\\" + mask;
      directories.pop();

      hFind = FindFirstFile(spec.c_str(), &ffd);
      if (hFind == INVALID_HANDLE_VALUE)  {
        return false;
      } 

      do {
        if (strcmp(ffd.cFileName, ".") != 0 && 
          strcmp(ffd.cFileName, "..") != 0) {
            if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
              directories.push(path + "\\" + ffd.cFileName);
            }
            else {
              files.push_back(path + "\\" + ffd.cFileName);
            }
        }
      } while (FindNextFile(hFind, &ffd) != 0);

      if (GetLastError() != ERROR_NO_MORE_FILES) {
        FindClose(hFind);
        return false;
      }

      FindClose(hFind);
      hFind = INVALID_HANDLE_VALUE;
    }

    return true;
  }
#endif
  bool IndDoublePredicate(const IndDouble& a, const IndDouble& b){return a.value > b.value;}

  void Haarby::init(int batch, string dn, bool binaryMode, int numF)
  {
    numFiles = numF;
    iterFiles = 0;
    storeBinary = binaryMode;
    outDn = dn;
    numBatch = batch;
    if (!thisBatchFeatures && storeBinary && numFiles > 0) 
    {
      thisBatchFeatures = new double[numBatch*numFiles];
    }
    if (!features) features = new double[numBatch];
    if (!featureIndex)
    {
      featureIndex = new int[HaarPts*HaarSize];
      int s = 0;
      for (int i=0; i < 20; i++)
      {
        for (int j=0; j < 40; j++)
        {
          for (int k=2; k <= 20-i; k+=2)
            for (int m=1; m <= 40-j; m++)
            {
              if (i+k-1 < 20 && j+m-1 < 40)
              {
                featureIndex[s*HaarPts+0]=j;
                featureIndex[s*HaarPts+1]=i;
                featureIndex[s*HaarPts+2]=j+m;
                featureIndex[s*HaarPts+3]=i;
                featureIndex[s*HaarPts+4]=j;
                featureIndex[s*HaarPts+5]=i+k/2;
                featureIndex[s*HaarPts+6]=j+m;
                featureIndex[s*HaarPts+7]=i+k/2;
                featureIndex[s*HaarPts+8]=j;
                featureIndex[s*HaarPts+9]=i+k;
                featureIndex[s*HaarPts+10]=j+m;
                featureIndex[s*HaarPts+11]=i+k;
                s++;
              } 
            }
            for (int k=2; k <= 40-j; k+=2)
              for (int m=1; m <= 20-i; m++)
              {
                if (i+m-1 < 20 && j+k-1 < 40)
                {
                  featureIndex[s*HaarPts+0]=j;
                  featureIndex[s*HaarPts+1]=i;
                  featureIndex[s*HaarPts+2]=j;
                  featureIndex[s*HaarPts+3]=i+m;
                  featureIndex[s*HaarPts+4]=j+k/2;
                  featureIndex[s*HaarPts+5]=i;
                  featureIndex[s*HaarPts+6]=j+k/2;
                  featureIndex[s*HaarPts+7]=i+m;
                  featureIndex[s*HaarPts+8]=j+k;
                  featureIndex[s*HaarPts+9]=i;
                  featureIndex[s*HaarPts+10]=j+k;
                  featureIndex[s*HaarPts+11]=i+m;
                  s++;
                }
              }
        }
      }
    }
  }

  void Haarby::setBatchIndex(int bI)
  {
    beginIndex = bI;
    stringstream ss;
    ss << outDn << "\\" << bI << "_" << numBatch;
    if (storeBinary) ss << ".bin";
    else ss << ".txt";
    ss>> outFn;
    iterFiles = 0;
  }

  void Haarby::getIntegralImage()
  {
    if (!integralImage) integralImage = new double[(pImage->width+1)*(pImage->height+1)];
    memset(integralImage,0,(pImage->width+1)*(pImage->height+1)*sizeof(double));
    if (!integralS) integralS = new double[(pImage->width+1)*(pImage->height+1)];
    memset(integralS,0,(pImage->width+1)*(pImage->height+1)*sizeof(double));

    for (int i=1; i < pImage->height+1; i++)
    {
      for (int j=1; j < pImage->width+1; j++)
      {
        CvScalar s = cvGet2D(pImage,i-1,j-1);
        integralS[i*(pImage->width+1)+j]=integralS[i*(pImage->width+1)+j-1]+s.val[0];
        integralImage[i*(pImage->width+1)+j]=
          integralImage[(i-1)*(pImage->width+1)+j]
          +integralS[i*(pImage->width+1)+j];
      }
    }
  }

  void Haarby::getFeatures(IplImage* image, int numFeatures, int* indFeatures, double* outFeatures)
  {
    pImage = image;
    getIntegralImage();
    for (int i=0; i<numFeatures; i++)
    {
      int x1=featureIndex[(indFeatures[i])*HaarPts+0];
      int y1=featureIndex[(indFeatures[i])*HaarPts+1];
      int x2=featureIndex[(indFeatures[i])*HaarPts+2];
      int y2=featureIndex[(indFeatures[i])*HaarPts+3];
      int x3=featureIndex[(indFeatures[i])*HaarPts+4];
      int y3=featureIndex[(indFeatures[i])*HaarPts+5];
      int x4=featureIndex[(indFeatures[i])*HaarPts+6];
      int y4=featureIndex[(indFeatures[i])*HaarPts+7];
      int x5=featureIndex[(indFeatures[i])*HaarPts+8];
      int y5=featureIndex[(indFeatures[i])*HaarPts+9];
      int x6=featureIndex[(indFeatures[i])*HaarPts+10];
      int y6=featureIndex[(indFeatures[i])*HaarPts+11];
      outFeatures[i] = -integralImage[y1*(HaarW+1)+x1] + integralImage[y2*(HaarW+1)+x2] 
      + 2*integralImage[y3*(HaarW+1)+x3] - 2*integralImage[y4*(HaarW+1)+x4] 
      - integralImage[y5*(HaarW+1)+x5] + integralImage[y6*(HaarW+1)+x6];
    }

  }

  void Haarby::calcFeatures(IplImage* image)
  {
    pImage = image;
    getIntegralImage();
    for (int i=0; i<numBatch; i++)
    {
      int x1=featureIndex[(beginIndex+i)*HaarPts+0];
      int y1=featureIndex[(beginIndex+i)*HaarPts+1];
      int x2=featureIndex[(beginIndex+i)*HaarPts+2];
      int y2=featureIndex[(beginIndex+i)*HaarPts+3];
      int x3=featureIndex[(beginIndex+i)*HaarPts+4];
      int y3=featureIndex[(beginIndex+i)*HaarPts+5];
      int x4=featureIndex[(beginIndex+i)*HaarPts+6];
      int y4=featureIndex[(beginIndex+i)*HaarPts+7];
      int x5=featureIndex[(beginIndex+i)*HaarPts+8];
      int y5=featureIndex[(beginIndex+i)*HaarPts+9];
      int x6=featureIndex[(beginIndex+i)*HaarPts+10];
      int y6=featureIndex[(beginIndex+i)*HaarPts+11];
      features[i] = -integralImage[y1*(HaarW+1)+x1] + integralImage[y2*(HaarW+1)+x2] 
        + 2*integralImage[y3*(HaarW+1)+x3] - 2*integralImage[y4*(HaarW+1)+x4] 
        - integralImage[y5*(HaarW+1)+x5] + integralImage[y6*(HaarW+1)+x6];
    }
    if (!storeBinary)
    {
      ofstream ff;
      ff.open(outFn.c_str(),ios_base::app);
      ff.flags(std::ios::fixed);
      ff.precision(0);
      for (int i=0; i < numBatch; i++)
      {
        ff << features[i] << " ";
      }
      ff << endl;
      ff.close();
    }
    else
    {
      memcpy(thisBatchFeatures+iterFiles*numBatch,features,numBatch*sizeof(double));
      iterFiles++;
    }
  }

  void Haarby::readBinaryFile(string fn, double* bMem)
  {
    ifstream file(fn.c_str(),ios::in|ios::binary|ios::ate);
    unsigned int size = file.tellg();
    file.seekg (0, ios::beg);
    file.read ((char*) &bMem[0], size);
    file.close();
  }

  void Haarby::saveBinaryFile()
  {
    ofstream ff;
    ff.open(outFn.c_str(),ios::binary);
    ff.write( (char*)&thisBatchFeatures[0], sizeof(double)*numFiles*numBatch ); 
    ff.close();
  }


  void Haarby::destroy()
  {
    if (thisBatchFeatures) delete[] thisBatchFeatures;
    if (features) delete[] features;
    if (featureIndex) delete[] featureIndex;
    if (integralImage) delete[] integralImage;
    if (integralS) delete[] integralS;
  }

}