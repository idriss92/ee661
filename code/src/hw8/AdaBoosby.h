#ifndef __ADABOOSBY_h__
#define __ADABOOSBY_h__

#include <iostream>
#include <string>
#include "cv.h"
#include "Haarby.h"

using namespace std;

namespace CVisby{

  class AdaBoosby
  {
  public:
    int partBatchNo;
    //Set the size of each batch feature
    void init(string fnDirec, int bSize, int numFeatures, int partBatch);
    //train a monolithic AdaBoost.
    void trainMono(vector<IndDouble>& errorT, vector<DecisionStump>& hT);
    //Evaluate the cascade classifier for the cascade loop, outerLoop = true: modify the negative training set.
    void evaluateCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT,
      const vector<int>& eachT, vector<double>& thresT, double* Fi, double* Di, Haarby& haarby, 
      double lastD, double d, int* PN, bool outerLoop);
    bool detectCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
      const vector<int>& eachT, vector<double>& thresT, double* features);
    //Do classification only with first tSize stages.
    bool detectCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
      const vector<int>& eachT, vector<double>& thresT, double* features, unsigned int tSize);
    //train a single stage AdaBoost in the cascaded loop
    void trainCascadeMono(vector<IndDouble>& errorT, vector<DecisionStump>& hT, int* PN, bool useSavedStage);
    //outer loop to train all the stages
    void trainCascade(double f, double d, double Ftarget, Haarby& haarby,string fnClassifierCascade);
    //test monolithic AdaBoost
    void testMono(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, Haarby& haarby);
    //test cascaded AdaBoost
    void testCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
      const vector<int>& eachT, vector<double>& thresT, Haarby& haarby);
    //void testCascadeDebug(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
    //  const vector<int>& eachT, vector<double>& thresT, Haarby& haarby);
    bool detectMono(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, double* features);
    void saveMonoFile(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT,string fnClassifierMono);
    void readMonoFile(vector<IndDouble>& errorT, vector<DecisionStump>& hT, string fnClassifierMono);
    void readCascadeFile(vector<IndDouble>& errorT, vector<DecisionStump>& hT, 
      vector<int>& eachT, vector<double>& thresT,
      string fnClassifierCas);
    void saveCascadeFile(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
      const vector<int>& eachT, const vector<double>& thresT,
      string fnClassifierCas);
    void outputCascade(const vector<IndDouble>& errorT, const vector<DecisionStump>& hT, 
      const vector<int>& eachT, const vector<double>& thresT);
  private:
    string fnDirectory;
    int batchSize;
    string fnPostive, fnNegative;
    vector<string> posfiles;
    vector<string> negfiles;
    int ADTIME;
    double* savedWeights;
  };

}

#endif