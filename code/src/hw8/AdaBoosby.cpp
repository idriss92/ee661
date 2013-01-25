#include "AdaBoosby.h"
#include "highgui.h"
#include "cv.h"
#include "cxcore.h"
#include "Haarby.h"
#include "time.h"
#include <fstream>

using namespace std;

namespace CVisby{

  void AdaBoosby::init(string fnDirec, int bSize, int numFeatures, int partBatch)
  {
    partBatchNo = partBatch;
    fnDirectory = fnDirec;
    batchSize = bSize;
    //get the directories for postive and negative image first
    if (fnDirectory.find("positive") != string::npos)
    {
      fnPostive = fnDirectory;
      fnNegative = fnDirectory.substr(0,fnDirectory.find("positive"));
      fnNegative += string("negative");
    }
    else if (fnDirectory.find("negative") != string::npos)
    {
      fnNegative = fnDirectory;
      fnPostive = fnDirectory.substr(0,fnDirectory.size()-fnDirectory.find("positive"));
      fnPostive += string("positive");
    }
    else
    {
      fnNegative = fnDirectory+string("\\negative");
      fnPostive = fnDirectory+string("\\positive");
    }
    //100 files for each 1660 batched features.
    //AdaBoost loop begins here
    ADTIME = numFeatures;
#ifdef _WIN32
    ListFiles(fnPostive, "*.png", posfiles);
    ListFiles(fnNegative, "*.png", negfiles);
#endif
  }

  void AdaBoosby::trainMono(vector<IndDouble>& errorT,vector<DecisionStump>& hT)
  {
    double* wts = new double[posfiles.size()+negfiles.size()];
    for (unsigned int i=0; i<posfiles.size(); i++)
      wts[i]=1.0/(2*posfiles.size());
    for (unsigned int i=0; i<negfiles.size(); i++)
      wts[i+posfiles.size()]=1.0/(2*negfiles.size());
    int bestFeature = -1;
    double bestError = 1;
    IndDouble* bestFeatureVector = new IndDouble[(posfiles.size()+negfiles.size())];
    vector<double> featureErrors;
    vector<DecisionStump> weakClassifiers;
    featureErrors.reserve(HaarSize);
    weakClassifiers.reserve(HaarSize);
    for (int t=1; t < ADTIME; t++)
    {
      clock_t t1 = clock();
      bestFeature = -1;
      bestError = 1;
      //Normalizing weights
      double wtssum = 0;
      for (unsigned int i=0; i < posfiles.size()+negfiles.size(); i++)
      {
        wtssum += wts[i];
      }
      for (unsigned int i=0; i < posfiles.size()+negfiles.size(); i++)
      {
        wts[i] = wts[i]/wtssum;
      }

      for (int i=0; i < min(partBatchNo,batchSize); i++)
      {
        clock_t tt1 = clock();
        //readin features first using txt
        /*stringstream ss;
        ss << i*(HaarSize/batchSize) << "_" << HaarSize/batchSize << ".txt";
        string fnIndex;
        ss >> fnIndex;
        string posiFile = fnPostive+"\\"+fnIndex;
        string negaFile = fnNegative+"\\"+fnIndex;
        IndDouble* columnFeatures = new IndDouble[HaarSize/batchSize * (posfiles.size()+negfiles.size())];
        ifstream fpPosi(posiFile.c_str());
        ifstream fpNega(negaFile.c_str());
        for (unsigned int numLine = 0; numLine < posfiles.size(); numLine++)
        {
          string line;
          getline(fpPosi,line);
          stringstream ssline(line);
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            ssline >> columnFeatures[j*(posfiles.size()+negfiles.size())+numLine].value;
            columnFeatures[j*(posfiles.size()+negfiles.size())+numLine].ind=numLine;
          }
        }
        for (unsigned int numLine = 0; numLine < negfiles.size(); numLine++)
        {
          string line;
          getline(fpNega,line);
          stringstream ssline(line);
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            ssline >> columnFeatures[j*(posfiles.size()+negfiles.size())+numLine+posfiles.size()].value;
            columnFeatures[j*(posfiles.size()+negfiles.size())+numLine+posfiles.size()].ind=numLine+posfiles.size();
          }
        }*/
        //reading features using binary file
        stringstream ss;
        ss << i*(HaarSize/batchSize) << "_" << HaarSize/batchSize << ".bin";
        string fnIndex;
        ss >> fnIndex;
        string posiFile = fnPostive+"\\"+fnIndex;
        string negaFile = fnNegative+"\\"+fnIndex;
        double* batchFeatures = new double[HaarSize/batchSize * (posfiles.size()+negfiles.size())];
        IndDouble* columnFeatures = new IndDouble[HaarSize/batchSize * (posfiles.size()+negfiles.size())];
        ifstream fpPosi(posiFile.c_str(),ios::in|ios::binary|ios::ate);
        ifstream fpNega(negaFile.c_str(),ios::in|ios::binary|ios::ate);
        unsigned int size = fpPosi.tellg();
        fpPosi.seekg (0, ios::beg);
        fpPosi.read ((char*) &batchFeatures[0], size);
        fpPosi.close();
        size = fpNega.tellg();
        fpNega.seekg (0, ios::beg);
        fpNega.read ((char*) &batchFeatures[HaarSize/batchSize * posfiles.size()], size);
        fpNega.close();
        for (unsigned int numLine = 0; numLine < posfiles.size(); numLine++)
        {
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            columnFeatures[j*(posfiles.size()+negfiles.size())+numLine].value=
              batchFeatures[numLine*(HaarSize/batchSize)+j];
            columnFeatures[j*(posfiles.size()+negfiles.size())+numLine].ind=numLine;
          }
        }
        for (unsigned int numLine = 0; numLine < negfiles.size(); numLine++)
        {
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            columnFeatures[j*(posfiles.size()+negfiles.size())+numLine+posfiles.size()].value
              =batchFeatures[(numLine+posfiles.size())*(HaarSize/batchSize)+j];
            columnFeatures[j*(posfiles.size()+negfiles.size())+numLine+posfiles.size()].ind=numLine+posfiles.size();
          }
        }
        delete[] batchFeatures;

        clock_t tt2p = clock();
        cout << "Time for reading features" << i << ":" << double(tt2p-tt1)/CLOCKS_PER_SEC << "s" << endl;

        IndDouble* sortedFeatureVector = new IndDouble[(posfiles.size()+negfiles.size())];
        double SumPosWeight = 0;
        double SumNegWeight = 0;
        for (unsigned int j=0; j < posfiles.size(); j++)
          SumPosWeight += wts[j];
        for (unsigned int j=posfiles.size(); j < posfiles.size()+negfiles.size(); j++)
          SumNegWeight += wts[j];
        //Get the batch features' weak classifiers
        for (int k=0; k < HaarSize/batchSize; k++)
        {
          IndDouble* pBegin = columnFeatures+k*(posfiles.size()+negfiles.size());
          IndDouble* pEnd = pBegin+(posfiles.size()+negfiles.size());
          memcpy(sortedFeatureVector,pBegin,sizeof(IndDouble)*(posfiles.size()+negfiles.size()));
          sort(sortedFeatureVector,sortedFeatureVector+(posfiles.size()+negfiles.size()),IndDoublePredicate);

          double SumPosWeightBelow = SumPosWeight;
          double SumNegWeightBelow = SumNegWeight;
          double possibleError = 1;
          double minError = 1;
          int polarity = 1;
          DecisionStump bestDS;
          for (unsigned int j=0; j < posfiles.size()+negfiles.size(); j++)
          {
            if ((unsigned)(sortedFeatureVector+j)->ind < posfiles.size() )
              SumPosWeightBelow -= wts[(sortedFeatureVector+j)->ind];
            else
              SumNegWeightBelow -= wts[(sortedFeatureVector+j)->ind];
            if (SumPosWeightBelow + SumNegWeight - SumNegWeightBelow < 
              SumNegWeightBelow + SumPosWeight - SumPosWeightBelow)
            {
              possibleError = SumPosWeightBelow + SumNegWeight - SumNegWeightBelow;
              polarity = -1;
            }
            else
            {
              possibleError = SumNegWeightBelow + SumPosWeight - SumPosWeightBelow;
              polarity = 1;
            }
            if (possibleError < minError)
            {
              bestDS.p = polarity;
              bestDS.theta = (sortedFeatureVector+j)->value-0.1;
              minError = possibleError;
            }
          }
          featureErrors.push_back(minError);
          weakClassifiers.push_back(bestDS);
          if (minError < bestError)
          {
            memcpy(bestFeatureVector,pBegin,sizeof(IndDouble)*(posfiles.size()+negfiles.size()));
            bestFeature = featureErrors.size()-1;
            bestError = minError;
          }
        }

        delete[] sortedFeatureVector;

        clock_t tt2 = clock();
        cout << "Time for Batch No." << i << ":" << double(tt2-tt1)/CLOCKS_PER_SEC << "s" << endl;

        delete[] columnFeatures;
      }
      //now the feature errors are in the vectors, as well as the weakClassifiers.
      //The best weak classifier has also been selected.
      int testF = 0;
      for (unsigned int i=0; i<posfiles.size(); i++)
      {
        int e = 0;
        if ( weakClassifiers[bestFeature].p * bestFeatureVector[i].value 
          < weakClassifiers[bestFeature].p * weakClassifiers[bestFeature].theta)
        {
          e=0;      
        }
        else {e=1;testF++;}
        wts[i]=wts[i]*(e==1?1:(bestError/(1-bestError)));
      }
      for (unsigned int i=posfiles.size(); i<negfiles.size()+posfiles.size(); i++)
      {
        int e = 0;
        if ( weakClassifiers[bestFeature].p * bestFeatureVector[i].value 
          < weakClassifiers[bestFeature].p * weakClassifiers[bestFeature].theta)
        {
          e=1;testF++;
        }
        else e=0;
        wts[i]=wts[i]*(e==1?1:(bestError/(1-bestError)));
      }
      IndDouble tmp;
      tmp.ind = bestFeature;
      tmp.value = bestError;
      errorT.push_back(tmp);
      hT.push_back(weakClassifiers[bestFeature]);

      clock_t t2 = clock();
      cout << "Time for AdaBoost Loop No." << t <<":" <<double(t2-t1)/CLOCKS_PER_SEC << "s" << endl;

      featureErrors.clear();
      weakClassifiers.clear();
    }
    delete[] bestFeatureVector;
    delete[] wts;
  }

  void AdaBoosby::trainCascadeMono(vector<IndDouble>& errorT,vector<DecisionStump>& hT, int* PN, bool useSavedStage)
  {
    unsigned int countFiles = 0;
    unsigned int countPos = 0;
    unsigned int countNeg = 0;
    for (unsigned int i=0; i < posfiles.size()+negfiles.size(); i++)
    {
      countFiles += abs(PN[i]);
      if (PN[i] == 1) ++countPos;
      if (PN[i] == -1) ++countNeg;
    }
    double* wts = new double[countFiles];
    if (!useSavedStage)
    {
      for (unsigned int i=0; i<countPos; i++)
        wts[i]=1.0/(2*countPos);
      for (unsigned int i=0; i<countNeg; i++)
        wts[i+countPos]=1.0/(2*countNeg);
    }
    else
    {
      for (unsigned int i=0; i<countFiles; i++)
      {
        wts[i] = savedWeights[i];
      }
    }
    int bestFeature = -1;
    double bestError = 1;
    IndDouble* bestFeatureVector = new IndDouble[countFiles];
    vector<double> featureErrors;
    vector<DecisionStump> weakClassifiers;
    featureErrors.reserve(HaarSize);
    weakClassifiers.reserve(HaarSize);
    for (int t=(useSavedStage? (ADTIME-1):1); t < ADTIME; t++)
    {
      clock_t t1 = clock();
      bestFeature = -1;
      bestError = 1;
      //Normalizing weights
      double wtssum = 0;
      for (unsigned int i=0; i < countFiles; i++)
      {
        wtssum += wts[i];
      }
      for (unsigned int i=0; i < countFiles; i++)
      {
        wts[i] = wts[i]/wtssum;
      }
      wtssum = 0;
      for (unsigned int i=0; i < countFiles; i++)
      {
        wtssum += wts[i];
      }

      for (int i=0; i < min(partBatchNo,batchSize); i++)
      {
        clock_t tt1 = clock();
        //readin features first using txt
        /*stringstream ss;
        ss << i*(HaarSize/batchSize) << "_" << HaarSize/batchSize << ".txt";
        string fnIndex;
        ss >> fnIndex;
        string posiFile = fnPostive+"\\"+fnIndex;
        string negaFile = fnNegative+"\\"+fnIndex;
        IndDouble* columnFeatures = new IndDouble[HaarSize/batchSize * (countFiles)];
        ifstream fpPosi(posiFile.c_str());
        ifstream fpNega(negaFile.c_str());
        int sampleIndex = 0;
        for (unsigned int numLine = 0; numLine < posfiles.size(); numLine++)
        {
          string line;
          getline(fpPosi,line);
          if ( PN[numLine] ==0) continue;
          
          stringstream ssline(line);
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            ssline >> columnFeatures[j*(countFiles)+sampleIndex].value;
            columnFeatures[j*(countFiles)+sampleIndex].ind=numLine;
          }
          sampleIndex ++;
        }
        for (unsigned int numLine = 0; numLine < negfiles.size(); numLine++)
        {
          string line;
          getline(fpNega,line);
          if ( PN[numLine+posfiles.size()] ==0) continue;
          
          stringstream ssline(line);
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            ssline >> columnFeatures[j*(countFiles)+sampleIndex].value;
            columnFeatures[j*(countFiles)+sampleIndex].ind=numLine+posfiles.size();
          }
          sampleIndex ++;
        }*/
        //readin features using binary
        stringstream ss;
        ss << i*(HaarSize/batchSize) << "_" << HaarSize/batchSize << ".bin";
        string fnIndex;
        ss >> fnIndex;
        string posiFile = fnPostive+"\\"+fnIndex;
        string negaFile = fnNegative+"\\"+fnIndex;
        int sampleIndex = 0;
        double* batchFeatures = new double[HaarSize/batchSize * (posfiles.size()+negfiles.size())];
        IndDouble* columnFeatures = new IndDouble[HaarSize/batchSize * (posfiles.size()+negfiles.size())];
        ifstream fpPosi(posiFile.c_str(),ios::in|ios::binary|ios::ate);
        ifstream fpNega(negaFile.c_str(),ios::in|ios::binary|ios::ate);
        unsigned int size = fpPosi.tellg();
        fpPosi.seekg (0, ios::beg);
        fpPosi.read ((char*) &batchFeatures[0], size);
        fpPosi.close();
        size = fpNega.tellg();
        fpNega.seekg (0, ios::beg);
        fpNega.read ((char*) &batchFeatures[HaarSize/batchSize * posfiles.size()], size);
        fpNega.close();
        for (unsigned int numLine = 0; numLine < posfiles.size(); numLine++)
        {
          if ( PN[numLine] ==0) continue;
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            columnFeatures[j*(countFiles)+sampleIndex].value=
              batchFeatures[numLine*(HaarSize/batchSize)+j];
            columnFeatures[j*(countFiles)+sampleIndex].ind=sampleIndex;//numLine;
          }
          sampleIndex++;
        }
        for (unsigned int numLine = 0; numLine < negfiles.size(); numLine++)
        {
          if ( PN[numLine+posfiles.size()] ==0) continue;
          for (int j=0; j < HaarSize/batchSize; j++)
          {
            columnFeatures[j*(countFiles)+sampleIndex].value
              =batchFeatures[(numLine+posfiles.size())*(HaarSize/batchSize)+j];
            columnFeatures[j*(countFiles)+sampleIndex].ind=sampleIndex;//numLine+posfiles.size();
          }
          sampleIndex++;
        }
        delete[] batchFeatures;
        IndDouble* sortedFeatureVector = new IndDouble[countFiles];
        double SumPosWeight = 0;
        double SumNegWeight = 0;
        for (unsigned int j=0; j < countPos; j++)
          SumPosWeight += wts[j];
        for (unsigned int j=countPos; j < countFiles; j++)
          SumNegWeight += wts[j];
        //Get the batch features' weak classifiers
        for (int k=0; k < HaarSize/batchSize; k++)
        {
          IndDouble* pBegin = columnFeatures+k*(countFiles);
          IndDouble* pEnd = pBegin+(countFiles);
          memcpy(sortedFeatureVector,pBegin,sizeof(IndDouble)*(countFiles));
          sort(sortedFeatureVector,sortedFeatureVector+(countFiles),IndDoublePredicate);

          double SumPosWeightBelow = SumPosWeight;
          double SumNegWeightBelow = SumNegWeight;
          double possibleError = 1;
          double minError = 1;
          int polarity = 1;
          DecisionStump bestDS;
          for (unsigned int j=0; j < countFiles; j++)
          {
            if (PN[(unsigned)(sortedFeatureVector+j)->ind] == 1 )
              SumPosWeightBelow -= wts[(sortedFeatureVector+j)->ind];
            else
              SumNegWeightBelow -= wts[(sortedFeatureVector+j)->ind];
            if (SumPosWeightBelow + SumNegWeight - SumNegWeightBelow < 
              SumNegWeightBelow + SumPosWeight - SumPosWeightBelow)
            {
              possibleError = SumPosWeightBelow + SumNegWeight - SumNegWeightBelow;
              polarity = -1;
            }
            else
            {
              possibleError = SumNegWeightBelow + SumPosWeight - SumPosWeightBelow;
              polarity = 1;
            }
            if (possibleError < minError)
            {
              bestDS.p = polarity;
              bestDS.theta = (sortedFeatureVector+j)->value-0.1;
              minError = possibleError;
            }
          }
          featureErrors.push_back(minError);
          weakClassifiers.push_back(bestDS);
          if (minError < bestError)
          {
            memcpy(bestFeatureVector,pBegin,sizeof(IndDouble)*(countFiles));
            bestFeature = featureErrors.size()-1;
            bestError = minError;
          }
        }

        delete[] sortedFeatureVector;
        delete[] columnFeatures;
      }
      //now the feature errors are in the vectors, as well as the weakClassifiers.
      //The best weak classifier has also been selected.
      int testF=0;
      for (unsigned int i=0; i<countPos; i++)
      {
        int e = 0;
        if ( weakClassifiers[bestFeature].p * bestFeatureVector[i].value 
          < weakClassifiers[bestFeature].p * weakClassifiers[bestFeature].theta)
        {
          e=0;      
        }
        else {e=1;testF++;}
        wts[i]=wts[i]*(e==1?1:(bestError/(1-bestError)));
      }
      for (unsigned int i=countPos; i<countFiles; i++)
      {
        int e = 0;
        if ( weakClassifiers[bestFeature].p * bestFeatureVector[i].value 
          < weakClassifiers[bestFeature].p * weakClassifiers[bestFeature].theta)
        {
          e=1;testF++;
        }
        else e=0;
        wts[i]=wts[i]*(e==1?1:(bestError/(1-bestError)));
      }
      IndDouble tmp;
      tmp.ind = bestFeature;
      tmp.value = bestError;
      errorT.push_back(tmp);
      hT.push_back(weakClassifiers[bestFeature]);

      clock_t t2 = clock();
      cout << "Time for AdaBoost Loop No." << t <<":" <<double(t2-t1)/CLOCKS_PER_SEC << "s" << endl;

      featureErrors.clear();
      weakClassifiers.clear();

      for (unsigned int i=0; i < countFiles; i++)
      {
        savedWeights[i] = wts[i];
      }
    }
    delete[] bestFeatureVector;
    delete[] wts;
  }

  void AdaBoosby::trainCascade(double f, double d, double Ftarget, Haarby& haarby, string fnClassifierCascade)
  {
    int *PN = new int[posfiles.size()+negfiles.size()];
    memset(PN,0,(posfiles.size()+negfiles.size())*sizeof(int));
    for (unsigned int i=0;i<posfiles.size(); i++)
      PN[i] = 1;
    for (unsigned int i=0;i<negfiles.size(); i++)
      PN[i+posfiles.size()] = -1;

    double Ft=1;
    double Dt=1;
    int iter = 0;
    
    vector<IndDouble> errorT;
    vector<DecisionStump> hT;
    vector<int> eachT;
    vector<double> thresT;

    bool breakOuter = false;
    while (Ft > Ftarget && !breakOuter)
    {
      int numNeg = 0;
      for (unsigned int i=posfiles.size(); i<posfiles.size()+negfiles.size(); i++)
        numNeg += (PN[i]==-1);
      cout << "Negative Number of Samples: " << numNeg << endl;
      if (numNeg == 0) 
      {
        breakOuter = true;
        break;
      }

      ++iter;
      int nt=0;
      double newF = 1;//Ft;
      savedWeights = new double[posfiles.size()+negfiles.size()];
      while (newF > f/**Ft*/)
      {
        if (nt > 0) 
        {
          eachT.pop_back();
          thresT.pop_back();
        }
        ++nt;
        ADTIME = nt+1;
        vector<IndDouble> errorTT;
        vector<DecisionStump> hTT;
        if (nt != 1) trainCascadeMono(errorTT,hTT,PN,true);
        else trainCascadeMono(errorTT,hTT,PN,false);
        for (unsigned int i=0; i < errorTT.size(); i++)
        {
          errorT.push_back(errorTT[i]);
          hT.push_back(hTT[i]);
        }
        eachT.push_back(nt);
        thresT.push_back(0.5);
        double Fi, Di;
        evaluateCascade(errorT,hT,eachT,thresT,&Fi,&Di, haarby, Dt, d, PN, false);
        
        newF = Fi;
        Dt = Di;
        cout << "Cascade Inner Loop No. " << nt << ": Fi: " << Fi << " Di: " << Di << endl;
      }
      delete [] savedWeights;

      double Fi, Di;
      evaluateCascade(errorT,hT,eachT,thresT,&Fi,&Di, haarby, Dt, d, PN, true);
      Ft = Fi;
      cout << "Cascade Loop No. " << iter << ": Ft: " << Ft << " Dt: " << Dt << endl;  
      outputCascade(errorT,hT,eachT,thresT);
      cout << "==============================================================================" << endl;
      saveCascadeFile(errorT,hT,eachT,thresT,fnClassifierCascade);
    }
    saveCascadeFile(errorT,hT,eachT,thresT,fnClassifierCascade);
    delete[] PN;
  }

  void AdaBoosby::readCascadeFile(vector<IndDouble>& errorT, vector<DecisionStump>& hT, 
    vector<int>& eachT, vector<double>& thresT,
    string fnClassifierCas)
  {
    ifstream fpInput(fnClassifierCas.c_str());
    string line;
    getline(fpInput,line);
    stringstream ss(line);
    unsigned int eachTSize;
    ss >> eachTSize;
    while (eachTSize > 0)
    {
      int eachTInt;
      double thres;
      ss >> eachTInt >> thres;
      eachT.push_back(eachTInt);
      thresT.push_back(thres);
      eachTSize--;
    }
    while(getline(fpInput,line))
    {
      stringstream ss(line);
      int ind;
      double value;
      int p;
      double theta;
      ss >> ind >> value >> p >> theta;
      IndDouble id;
      DecisionStump ds;
      id.ind = ind;
      id.value = value;
      ds.p = p;
      ds.theta = theta;
      errorT.push_back(id);
      hT.push_back(ds);
    }
  }

  void AdaBoosby::saveCascadeFile(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
    const vector<int>& eachT, const vector<double>& thresT,
    string fnClassifierCas)
  {
    ofstream binClassifierMono(fnClassifierCas.c_str());
    binClassifierMono.flags(std::ios::fixed);
    binClassifierMono.precision(17);  
    binClassifierMono << eachT.size() << " ";
    for (unsigned int i=0; i < eachT.size(); i++)
    {
      binClassifierMono << eachT[i] << " " << thresT[i] << " ";
    }
    binClassifierMono<< endl;
    for (unsigned int i=0; i < errorT.size(); i++)
    {
      binClassifierMono << errorT[i].ind << " " << errorT[i].value << " "
        << hT[i].p << " " << hT[i].theta << endl;
    }
    binClassifierMono.close();
  }

  void AdaBoosby::outputCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
    const vector<int>& eachT, const vector<double>& thresT)
  {
    cout << "No. of Cascades: " << eachT.size();
    unsigned int beginT = 0;
    for (unsigned int i=0; i < eachT.size(); i++)
    {
      cout << "Cascade No. " << i << ": " << eachT[i] << " features" << endl;
      cout << "Threshold: " << thresT[i] << endl;
      cout << "ind errv p theta" << endl;
      for (unsigned int j=beginT; j < beginT+eachT[i]; j++)
      {
        cout << errorT[j].ind << " " << errorT[j].value << " "
          << hT[j].p << " " << hT[j].theta << endl;
      }
      beginT += eachT[i];
    }
  }

  void AdaBoosby::evaluateCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT,
    const vector<int>& eachT, vector<double>& thresT, double* Fi, double* Di, Haarby& haarby, 
    double lastD, double d, int* PN, bool outerLoop)
  {
    *Fi = 1;
    *Di = 1;
    int* indFeatures = new int[errorT.size()];
    for (unsigned int i=0; i < errorT.size(); i++)
    {
      indFeatures[i] = errorT[i].ind;
    }
    double* allFileFeatures = new double[errorT.size()*(posfiles.size()+negfiles.size())];
    int fileIndex = 0;
    for (vector<string>::iterator it = posfiles.begin(); it != posfiles.end(); ++it) {
      IplImage* pImage = cvLoadImage(it->c_str());
      IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
      cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);
      double* outFeatures = allFileFeatures+fileIndex*(errorT.size());
      for (unsigned int i=0; i < errorT.size(); i++)
      {
        outFeatures[i] = 0;
      }
      haarby.getFeatures(pImageGray,errorT.size(),indFeatures, outFeatures);
      cvReleaseImage(&pImage);
      cvReleaseImage(&pImageGray);
      fileIndex++;
    }
    for (vector<string>::iterator it = negfiles.begin(); it != negfiles.end(); ++it) {
      IplImage* pImage = cvLoadImage(it->c_str());
      IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
      cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);

      double* outFeatures = allFileFeatures+fileIndex*(errorT.size());
      for (unsigned int i=0; i < errorT.size(); i++)
      {
        outFeatures[i] = 0;
      }
      haarby.getFeatures(pImageGray,errorT.size(),indFeatures, outFeatures);
      //if(!detectMono(errorT,hT,outFeatures)) numCorrect++;
      cvReleaseImage(&pImage);
      cvReleaseImage(&pImageGray);
      fileIndex++;
    }
    *Di=0;
    if (!outerLoop) thresT[thresT.size()-1]=0.51;
    while (*Di < d /** lastD*/ && !outerLoop)
    {
      int truePositive = 0;
      int falsePositive = 0;
      thresT[thresT.size()-1]-=0.01;
      int testF = 0;
      int worstFile = -1;
      for (unsigned int j=0; j < posfiles.size(); j++)
      {
        double* fileFeatures = allFileFeatures+j*(errorT.size());
        double sumAlpha = 0;
        double sumAlphaH = 0;
        int featureBegin = 0;
        for (unsigned int i=0; i < eachT.size()-1; i++)
          featureBegin+= eachT[i];
        for (unsigned int i=0; (int) i<eachT[eachT.size()-1]; i++)
        {
          double beta = errorT[featureBegin+i].value/(1-errorT[featureBegin+i].value);
          double alpha = - log(beta);
          if ( alpha > DBL_MAX || alpha < -DBL_MAX) alpha = CVisby::AlphaINF;
          sumAlpha += alpha;
          int ht = (hT[featureBegin+i].p * fileFeatures[featureBegin+i] < hT[featureBegin+i].p * hT[featureBegin+i].theta) ? 1 : 0;
          sumAlphaH += alpha * ht;
        }
        if (sumAlphaH < 0.5 * sumAlpha) testF++;
        if (sumAlphaH/sumAlpha < thresT[thresT.size()-1]) 
        {
          thresT[thresT.size()-1] = sumAlphaH/sumAlpha;
          worstFile = j;
        }
      }
      //if (eachT[eachT.size()-1] >5) cout << "Worst Pos File: " << worstFile << endl; 
      int countNeg = 0;
      for (unsigned int j=posfiles.size(); j < posfiles.size()+negfiles.size(); j++)
      {
        if (PN[j] != -1) continue;
        else countNeg ++;
        double* fileFeatures = allFileFeatures+j*(errorT.size());
        double sumAlpha = 0;
        double sumAlphaH = 0;
        int featureBegin = 0;
        for (unsigned int i=0; i < eachT.size()-1; i++)
          featureBegin+= eachT[i];
        for (unsigned int i=0; (int) i<eachT[eachT.size()-1]; i++)
        {
          double beta = errorT[featureBegin+i].value/(1-errorT[featureBegin+i].value);
          double alpha = - log(beta);
          if ( alpha > DBL_MAX || alpha < -DBL_MAX) alpha = CVisby::AlphaINF;
          sumAlpha += alpha;
          int ht = (hT[featureBegin+i].p * fileFeatures[featureBegin+i] < hT[featureBegin+i].p * hT[featureBegin+i].theta) ? 1 : 0;
          sumAlphaH += (ht==0)? 0 : alpha;
        }
        if (sumAlphaH < thresT[thresT.size()-1] * sumAlpha) 
          ;
        else
          falsePositive++;
      }
      *Fi = double(falsePositive) / countNeg;
      *Di = 1;//double(truePositive) / (posfiles.size());
      cout << "Cascade Evaluate Loop No. " << (int)((0.5-thresT[thresT.size()-1])/0.01) << ": Fi: " << *Fi << " Di: " << *Di << endl;
    }
    if (outerLoop)
    {
      int truePositive = 0;
      int falsePositive = 0;
      for (unsigned int i=0; i < posfiles.size(); i++)
      {
        double* fileFeatures = allFileFeatures+i*(errorT.size());
        if (detectCascade(errorT,hT,eachT,thresT,fileFeatures))
          truePositive++;
      }
      for (unsigned int i=posfiles.size(); i < posfiles.size()+negfiles.size(); i++)
      {
        double* fileFeatures = allFileFeatures+i*(errorT.size());
        if (detectCascade(errorT,hT,eachT,thresT,fileFeatures))
          falsePositive++;
        else
        {
          PN[i]=0;
        }
        
      }
      *Fi = double(falsePositive) / (negfiles.size());
      *Di = double(truePositive) / (posfiles.size());
    }
    delete[] indFeatures;
    delete[] allFileFeatures;
  }

  bool AdaBoosby::detectCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
    const vector<int>& eachT, vector<double>& thresT, double* features)
  {
    int featureBegin = 0;
    for (unsigned j=0; j<eachT.size(); j++)
    {
      int featureEnd = featureBegin + eachT[j];
      double sumAlpha = 0;
      double sumAlphaH = 0;
      for (unsigned int i=0; (int) i<eachT[j]; i++)
      {
        double beta = errorT[featureBegin+i].value/(1-errorT[featureBegin+i].value);
        double alpha = - log(beta);
        if ( alpha > DBL_MAX || alpha < -DBL_MAX) alpha = CVisby::AlphaINF;
        sumAlpha += alpha;
        int ht = (hT[featureBegin+i].p * features[featureBegin+i] < hT[featureBegin+i].p * hT[featureBegin+i].theta) ? 1 : 0;
        sumAlphaH += alpha * ht;
      }
      if (sumAlphaH >= (thresT[j])*sumAlpha)
        ;
      else return false;
      featureBegin += eachT[j];
    }
    return true;
  }

  bool AdaBoosby::detectCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
    const vector<int>& eachT, vector<double>& thresT, double* features, unsigned int tSize)
  {
    int featureBegin = 0;
    for (unsigned j=0; j<tSize; j++)
    {
      int featureEnd = featureBegin + eachT[j];
      double sumAlpha = 0;
      double sumAlphaH = 0;
      for (unsigned int i=0; (int) i<eachT[j]; i++)
      {
        double beta = errorT[featureBegin+i].value/(1-errorT[featureBegin+i].value);
        double alpha = - log(beta);
        if ( alpha > DBL_MAX || alpha < -DBL_MAX) alpha = CVisby::AlphaINF;
        sumAlpha += alpha;
        int ht = (hT[featureBegin+i].p * features[featureBegin+i] < hT[featureBegin+i].p * hT[featureBegin+i].theta) ? 1 : 0;
        sumAlphaH += alpha * ht;
      }
      if (sumAlphaH >= (thresT[j])*sumAlpha)
        ;
      else return false;
      featureBegin += eachT[j];
    }
    return true;
  }

  void AdaBoosby::saveMonoFile(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, string fnClassifierMono)
  {
    ofstream binClassifierMono(fnClassifierMono.c_str());
    binClassifierMono.flags(std::ios::fixed);
    binClassifierMono.precision(6);  
    for (unsigned int i=0; i < errorT.size(); i++)
    {
      binClassifierMono << errorT[i].ind << " " << errorT[i].value << " "
        << hT[i].p << " " << hT[i].theta << endl;
    }
    binClassifierMono.close();
  }

  void AdaBoosby::readMonoFile(vector<IndDouble>& errorT, vector<DecisionStump>& hT, string fnClassifierMono)
  {
    ifstream fpInput(fnClassifierMono.c_str());
    string line;
    while(getline(fpInput,line))
    {
      stringstream ss(line);
      int ind;
      double value;
      int p;
      double theta;
      ss >> ind >> value >> p >> theta;
      IndDouble id;
      DecisionStump ds;
      id.ind = ind;
      id.value = value;
      ds.p = p;
      ds.theta = theta;
      errorT.push_back(id);
      hT.push_back(ds);
    }
  }

  bool AdaBoosby::detectMono(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, double* features)
  {
    double sumAlpha = 0;
    double sumAlphaH = 0;
    for (unsigned int i=0; i<errorT.size(); i++)
    {
      double beta = errorT[i].value/(1-errorT[i].value);
      double alpha = - log(beta);
      if ( alpha > DBL_MAX || alpha < -DBL_MAX) alpha = CVisby::AlphaINF;
      sumAlpha += alpha;
      int ht = (hT[i].p * features[i] < hT[i].p * hT[i].theta) ? 1 : 0;
      sumAlphaH += alpha * ht;
    }
    return (sumAlphaH >= 0.5*sumAlpha);
  }

  void AdaBoosby::testMono(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, Haarby& haarby)
  {
    int numCorrect = 0;
    int truePositive = 0;
    int falsePositive = 0;
    for (vector<string>::iterator it = posfiles.begin(); it != posfiles.end(); ++it) {
      IplImage* pImage = cvLoadImage(it->c_str());
      IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
      cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);

      int* indFeatures = new int[errorT.size()];
      double* outFeatures = new double[errorT.size()];
      for (unsigned int i=0; i < errorT.size(); i++)
      {
        indFeatures[i] = errorT[i].ind;
        outFeatures[i] = 0;
      }
      haarby.getFeatures(pImageGray,ADTIME-1,indFeatures, outFeatures);
      if(detectMono(errorT,hT,outFeatures)) 
      {
        numCorrect++;
        truePositive++;
      }
      delete[] indFeatures;
      delete[] outFeatures;
      cvReleaseImage(&pImage);
      cvReleaseImage(&pImageGray);
    }
    for (vector<string>::iterator it = negfiles.begin(); it != negfiles.end(); ++it) {
      IplImage* pImage = cvLoadImage(it->c_str());
      IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
      cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);

      int* indFeatures = new int[errorT.size()];
      double* outFeatures = new double[errorT.size()];
      for (unsigned int i=0; i < errorT.size(); i++)
      {
        indFeatures[i] = errorT[i].ind;
        outFeatures[i] = 0;
      }
      haarby.getFeatures(pImageGray,ADTIME-1,indFeatures, outFeatures);
      if(!detectMono(errorT,hT,outFeatures)) numCorrect++;
      else falsePositive ++;
      cvReleaseImage(&pImage);
      cvReleaseImage(&pImageGray);
    }
    ofstream ffTest("testMono.txt");
    ffTest << "Num Correct: " << numCorrect << " Total: " << posfiles.size()+negfiles.size() 
      << " True Positive: " << truePositive << " False Positive: " << falsePositive << endl;
    ffTest.close();
  }

  void AdaBoosby::testCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
    const vector<int>& eachT, vector<double>& thresT, Haarby& haarby)
  {
    for (unsigned int tSize = 1; tSize <= eachT.size(); tSize++)
    {
      int numCorrect = 0;
      int truePositive = 0;
      int falsePositive = 0;
      for (vector<string>::iterator it = posfiles.begin(); it != posfiles.end(); ++it) {
        IplImage* pImage = cvLoadImage(it->c_str());
        IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
        cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);

        int* indFeatures = new int[errorT.size()];
        double* outFeatures = new double[errorT.size()];
        for (unsigned int i=0; i < errorT.size(); i++)
        {
          indFeatures[i] = errorT[i].ind;
          outFeatures[i] = 0;
        }
        haarby.getFeatures(pImageGray,errorT.size(),indFeatures, outFeatures);
        if(detectCascade(errorT,hT,eachT,thresT,outFeatures,tSize)) 
        {
          numCorrect++;
          truePositive++;
        }
        delete[] indFeatures;
        delete[] outFeatures;
        cvReleaseImage(&pImage);
        cvReleaseImage(&pImageGray);
      }
      for (vector<string>::iterator it = negfiles.begin(); it != negfiles.end(); ++it) {
        IplImage* pImage = cvLoadImage(it->c_str());
        IplImage* pImageGray = cvCreateImage(cvGetSize(pImage),8,1);
        cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);

        int* indFeatures = new int[errorT.size()];
        double* outFeatures = new double[errorT.size()];
        for (unsigned int i=0; i < errorT.size(); i++)
        {
          indFeatures[i] = errorT[i].ind;
          outFeatures[i] = 0;
        }
        haarby.getFeatures(pImageGray,errorT.size(),indFeatures, outFeatures);
        if(!detectCascade(errorT,hT,eachT,thresT,outFeatures,tSize)) numCorrect++;
        else falsePositive ++;
        cvReleaseImage(&pImage);
        cvReleaseImage(&pImageGray);
      } 
      ofstream ffTest("testCascade.txt",ios::app);
      ffTest << "Num Correct: " << numCorrect << " Total: " << posfiles.size()+negfiles.size() 
        << " True Positive: " << truePositive << " False Positive: " << falsePositive << endl;
      ffTest.close();
    }
    
  }

  //void AdaBoosby::testCascadeDebug(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
  //  const vector<int>& eachT, vector<double>& thresT, Haarby& haarby)
  //{
  //  int *PN = new int[posfiles.size()+negfiles.size()];
  //  memset(PN,0,(posfiles.size()+negfiles.size())*sizeof(int));
  //  for (unsigned int i=0;i<posfiles.size(); i++)
  //    PN[i] = 1;
  //  for (unsigned int i=0;i<negfiles.size(); i++)
  //    PN[i+posfiles.size()] = -1;

  //  double f = 0.3;
  //  double d = 0.995;
  //  double Ftarget = 0.0000001;
  //  double Ft=1;
  //  double Dt=1;
  //  int iter = 0;
  //  int featureBegin = 0;
  //  
  //  vector<IndDouble> errorTDebug;
  //  vector<DecisionStump> hTDebug;
  //  vector<int> eachTDebug;
  //  vector<double> thresTDebug;

  //  bool breakOuter = false;
  //  while (Ft > Ftarget && !breakOuter)
  //  {
  //    int numNeg = 0;
  //    for (unsigned int i=posfiles.size(); i<posfiles.size()+negfiles.size(); i++)
  //      numNeg += (PN[i]==-1);
  //    cout << "Negative Number of Samples: " << numNeg << endl;
  //    if (numNeg == 0) 
  //    {
  //      breakOuter = true;
  //      break;
  //    }

  //    ++iter;
  //    int nt=0;
  //    double newF = 1;//Ft;
  //    //savedWeights = new double[posfiles.size()+negfiles.size()];
  //    savedWeights = new double[posfiles.size()+negfiles.size()];
  //    while (newF > f/**Ft*/)
  //    {
  //      /*if (iter != 7)
  //      {*/
  //        for (int i=0; i<eachT[iter-1]; i++)
  //        {
  //          errorTDebug.push_back(errorT[featureBegin+i]);
  //          hTDebug.push_back(hT[featureBegin+i]);
  //        }
  //        thresTDebug.push_back(thresT[iter-1]);
  //        eachTDebug.push_back(eachT[iter-1]);
  //        double Fi, Di;
  //        evaluateCascade(errorTDebug,hTDebug,eachTDebug,thresTDebug,&Fi,&Di,haarby,Dt,d,PN,false);
  //        newF = Fi;
  //      //}
  //      //else
  //      //{
  //      //  if (nt > 0) 
  //      //  {
  //      //    eachTDebug.pop_back();
  //      //    thresTDebug.pop_back();
  //      //  }
  //      //  /*for (int k=0;k<nt;k++)
  //      //  {
  //      //    errorT.pop_back();
  //      //    hT.pop_back();
  //      //  }*/
  //      //  ++nt;
  //      //  //********************************
  //      //  ADTIME = nt+1;
  //      //  vector<IndDouble> errorTT;
  //      //  vector<DecisionStump> hTT;
  //      //  if (nt != 1) trainCascadeMono(errorTT,hTT,PN,true);
  //      //  else trainCascadeMono(errorTT,hTT,PN,false);
  //      //  //************************************//
  //      //  for (unsigned int i=0; i < errorTT.size(); i++)
  //      //  {
  //      //    errorTDebug.push_back(errorTT[i]);
  //      //    hTDebug.push_back(hTT[i]);
  //      //  }
  //      //  eachTDebug.push_back(nt);
  //      //  thresTDebug.push_back(0.5);
  //      //  double Fi, Di;
  //      //  evaluateCascade(errorTDebug,hTDebug,eachTDebug,thresTDebug,&Fi,&Di, haarby, Dt, d, PN, false);
  //      //  newF = Fi;
  //      //}
  //    }
  //    delete[] savedWeights;
  //    featureBegin+=eachT[iter-1];
  //    //delete [] savedWeights;

  //    double Fi, Di;
  //    evaluateCascade(errorTDebug,hTDebug,eachTDebug,thresTDebug,&Fi,&Di, haarby, Dt, d, PN, true); // modify the training set.
  //    Ft = Fi;
  //    cout << "Cascade Loop No. " << iter << ": Ft: " << Ft << " Dt: " << Dt << endl;  
  //    outputCascade(errorT,hT,eachT,thresT);
  //    cout << "==============================================================================" << endl;
  //    //saveCascadeFile(errorT,hT,eachT,thresT,fnClassifierCascade);
  //  }
  //  //saveCascadeFile(errorT,hT,eachT,thresT,fnClassifierCascade);
  //  delete[] PN;
  //  /*ofstream ffTest("testCascade.txt");
  //  ffTest << "Num Correct: " << numCorrect << " Total: " << posfiles.size()+negfiles.size() 
  //    << " True Positive: " << truePositive << " False Positive: " << falsePositive << endl;
  //  ffTest.close();*/
  //}

}