#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>
#include <string>
#include <vector>
#include "Haarby.h"
#include "time.h"
#include "AdaBoosby.h"

using namespace CVisby;
using namespace std;

int usage(char **argv)
{
  std::string filename(argv[0]);
#ifdef _WIN32
  filename = filename.substr( filename.find_last_of( '\\' ) +1 );
#else
  filename = filename.substr( filename.find_last_of( '/' ) +1 );
#endif
  std::cout << "Usage: " << filename << " <-dXX> [-c] [-ADCXX] [-ADMXX] [-tmXX] [-tcXX]\n\t<-dXX> \
    The directory which contains the positive image set, and in the same level directory, there is another similar named negative image set.\n\t[-c]\
    Calculate the haar features and store to binary files in the same image directory.\n\t[-ADCXX]\
    Train the cacaded AdaBoost and write the result to XX.\n\t[-ADMXX]\
    Train the monolithic AdaBoost and write the result to XX.\n\t[-tmXX]\
    Read in the monolithic AdaBoost from XX and test it on the images in the directory set using -d parameter.\n\t[-tcXX]\
    Read in the cascaded AdaBoost from XX and test it on the images in the directory set using -d parameter.\n\t";
  return 1;
}

int main(int argc, char **argv)
{
  string fnDirectory;
  bool calcFeatures = false;
  string fnFeatures;
  bool testMono = false;
  bool testCascade = false;
  bool adaBoostMono = false;
  string fnClassifierMono;
  bool adaBoostCascade = false;
  string fnClassifierCas;

  if ( argc >= 2 )
  {
    for (int i=1; i < argc; i++)
    {
      string argi=string(argv[i]);
      if (argi.find("-d") != string::npos)
      {
        argi = argi.substr(argi.find("-d")+2);
        stringstream ss(argi);
        ss >> fnDirectory;
      }
      else if (argi.find("-c") != string::npos)
      {
        argi = argi.substr(argi.find("-c")+2);
        stringstream ss(argi);
        ss >> fnFeatures;
        calcFeatures = true;
      }
      else if (argi.find("-tm") != string::npos)
      {
        argi = argi.substr(argi.find("-tm")+3);
        stringstream ss(argi);
        ss >> fnClassifierMono;
        testMono = true;
      }
      else if (argi.find("-tc") != string::npos)
      {
        argi = argi.substr(argi.find("-tc")+3);
        stringstream ss(argi);
        ss >> fnClassifierCas;
        testCascade = true;
      }
      else if (argi.find("-ADM") != string::npos)
      {
        argi = argi.substr(argi.find("-ADM")+4);
        stringstream ss(argi);
        ss >> fnClassifierMono;
        adaBoostMono= true;
      }
      else if (argi.find("-ADC") != string::npos)
      {
        argi = argi.substr(argi.find("-ADC")+4);
        stringstream ss(argi);
        ss >> fnClassifierCas;
        adaBoostCascade = true;
      }
    }
  }
  else
  {
    return usage(argv);
  }

  int batchSize = 20;
  if (calcFeatures)
  {
    cout << "Calculating Haar Features and Storing in Files" << endl;
    clock_t t1 = clock();
    vector<string> files;
#ifdef _WIN32
    bool imagesReadable = ListFiles(fnDirectory, "*.png", files);
#endif 
    Haarby haarby;
    haarby.init(HaarSize/batchSize,fnDirectory, true, files.size());
    if (imagesReadable) {
      for (int i=0; i < batchSize; i++)
      {
        haarby.setBatchIndex(i*(HaarSize/batchSize));
        for (vector<string>::iterator it = files.begin(); it != files.end(); ++it) {
          IplImage* pImage = cvLoadImage(it->c_str());
          IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
          cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);       
          haarby.calcFeatures(pImageGray);
          cvReleaseImage(&pImage);
          cvReleaseImage(&pImageGray);
        }
        haarby.saveBinaryFile();
      }
    }
    haarby.destroy();
    clock_t t2 = clock();
    cout << "Time for feature calculation:" << double(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
  }

  if (adaBoostMono)
  {
    AdaBoosby adaboosby;
    adaboosby.init(fnDirectory, batchSize, 200, batchSize); // 3,2
    vector<IndDouble> errorT;
    vector<DecisionStump> hT;
    adaboosby.trainMono(errorT,hT);
    adaboosby.saveMonoFile(errorT,hT,fnClassifierMono);
  }
  if (adaBoostCascade)
  {
    Haarby haarby;
    haarby.init(HaarSize/batchSize,fnDirectory);
    AdaBoosby adaBoosby;
    adaBoosby.init(fnDirectory, batchSize, 200, batchSize); // 3,2
    adaBoosby.trainCascade(0.3, 0.99, 0.0000001, haarby, fnClassifierCas);
    haarby.destroy();
  }

  if (testMono)
  {
    Haarby haarby;
    haarby.init(HaarSize/batchSize,fnDirectory);
    AdaBoosby adaBoosby;
    adaBoosby.init(fnDirectory, batchSize, 200, batchSize); // 3,2
    vector<IndDouble> errorT;
    vector<DecisionStump> hT;
    adaBoosby.readMonoFile(errorT,hT,fnClassifierMono);
    adaBoosby.testMono(errorT,hT,haarby);
    haarby.destroy();
  }

  if (testCascade)
  {
    Haarby haarby;
    haarby.init(HaarSize/batchSize,fnDirectory);
    AdaBoosby adaBoosby;
    adaBoosby.init(fnDirectory, batchSize, 200, batchSize); // 3,2
    vector<IndDouble> errorT;
    vector<DecisionStump> hT;
    vector<int> eachT;
    vector<double> thresT;
    adaBoosby.readCascadeFile(errorT,hT,eachT,thresT,fnClassifierCas);
    adaBoosby.testCascade(errorT,hT,eachT,thresT,haarby);
    haarby.destroy();
  }

}
