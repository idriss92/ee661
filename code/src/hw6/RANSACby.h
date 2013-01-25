#ifndef __RANSACBY_h__
#define __RANSACBY_h__

#include <cv.h>
#include <vector>
#include "harrisby.h"

using namespace std;

namespace CVisby{

  const int CVisbyINF = 2147483647;
  const int RANSAC_SAMPLESIZE = 4;
  const double RANSAC_INLIERTHRES = 10; // set to large for hw6, originally=5
  class RANSACby{
  public:
    void runRANSAC(
      const vector<HarrisCorner>& corners1, const vector<HarrisCorner>& corners2,
      const vector<pair<pair<int,int>,double> >& pairs, bool useRoulette, double* returnH,
      vector<pair<pair<int,int>,double> >& pairs_Inlier);
    //H: transform image1 to image2. x_image2 = H * x_image1, H should be an already allocated 3*3 matrix.
    //The return value H is row-priority.
    int RANSACby::countInliers(const vector<HarrisCorner>& corners1, 
      const vector<HarrisCorner>& corners2, const vector<pair<pair<int,int>,double> >& pairs, 
      double* H1ToH2, double marginThres, unsigned int* inliers = NULL);
    //This function returns the number of inliers among the pairs.
    void randomSample(const vector<pair<pair<int,int>,double> >& pairs,unsigned int* randResult, int len);
    void randomWeightedSample(const vector<pair<pair<int,int>,double> >& pairs,unsigned int* randResult, int len);
  };

}

#endif
