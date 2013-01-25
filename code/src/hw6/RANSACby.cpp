#include "RANSACby.h"
#include "utiliby.h"
#include "homograby.h"
#include <iostream>
using namespace std;

#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <time.h>
#include <cstdlib>

namespace CVisby{
  
  void RANSACby::runRANSAC( 
    const vector<HarrisCorner>& corners1, const vector<HarrisCorner>& corners2, 
    const vector<pair<pair<int,int>,double>>& pairs, bool useRoulette, double* returnH,
    vector<pair<pair<int,int>,double>>& pairs_Inlier)
  {
    srand( (unsigned int)time(NULL));
    int n = CVisbyINF;
    int sample_count = 0;
    double* bestH1ToH2 = new double[9];
    int bestSupport = -1;
    while( n > sample_count)
    {
      unsigned int * sample = new unsigned int[RANSAC_SAMPLESIZE];
      memset(sample, pairs.size(), RANSAC_SAMPLESIZE*sizeof(unsigned int));
      if (useRoulette)
        randomWeightedSample(pairs,sample,RANSAC_SAMPLESIZE);
      else 
        randomSample(pairs,sample,RANSAC_SAMPLESIZE);
      double *x = new double[RANSAC_SAMPLESIZE];
      double *y = new double[RANSAC_SAMPLESIZE];
      double *xp = new double[RANSAC_SAMPLESIZE];
      double *yp = new double[RANSAC_SAMPLESIZE];
      for (int i=0; i < RANSAC_SAMPLESIZE; i++)
      {
        int c1 = pairs[sample[i]].first.first;
        int c2 = pairs[sample[i]].first.second;
        x[i] = corners1[(unsigned int) c1].x;
        y[i] = corners1[(unsigned int) c1].y;
        xp[i] = corners2[(unsigned int) c2].x;
        yp[i] = corners2[(unsigned int) c2].y;
      }  
      Homograby homography(3, RANSAC_SAMPLESIZE, x, y, xp, yp);
      double *H1ToH2 = new double[homography.HomoDim+1];
      homography.calcH_BDLT(H1ToH2,1);
      Utiliby::printMatrix(H1ToH2,3,3,0);
      int support = countInliers(corners1,corners2,pairs, H1ToH2, RANSAC_INLIERTHRES);

      if (support > bestSupport)
      {
        memcpy(bestH1ToH2, H1ToH2, (homography.HomoDim+1)*sizeof(double));
        bestSupport = support;
        cout << "New Best Support: " << bestSupport << endl;
      }

      cout << sample_count << ":" << n << endl;

      double epsilon = 1 - double(support) / pairs.size();
      int tmp = (int)(log(0.01)/log(1-pow(1-epsilon,RANSAC_SAMPLESIZE)));
      n = tmp < 0 ? n : tmp;
      ++sample_count;
      delete[] sample;
      delete[] x;
      delete[] y;
      delete[] xp;
      delete[] yp;
      delete H1ToH2;
    }

    //Now use the DLT algorithm to computer a better H
    unsigned int* inliers = new unsigned int[pairs.size()];
    //this is a redundant step;
    bestSupport = countInliers(corners1, corners2, pairs, bestH1ToH2, RANSAC_INLIERTHRES, inliers);
    double *x = new double[bestSupport];
    double *y = new double[bestSupport];
    double *xp = new double[bestSupport];
    double *yp = new double[bestSupport];
    for (int i=0; i < bestSupport; i++)
    {
      int c1 = pairs[inliers[i]].first.first;
      int c2 = pairs[inliers[i]].first.second;
      pairs_Inlier.push_back(std::pair<std::pair<int,int>,double>(std::pair<int,int>(c1,c2),pairs[inliers[i]].second));
      x[i] = corners1[(unsigned int) c1].x;
      y[i] = corners1[(unsigned int) c1].y;
      xp[i] = corners2[(unsigned int) c2].x;
      yp[i] = corners2[(unsigned int) c2].y;
    }  
    Homograby homography(3, bestSupport, x, y, xp, yp);
    double *dltH1ToH2 = new double[homography.HomoDim+1];
    homography.calcH_DLT(dltH1ToH2,1);
    if (returnH != NULL)
      memcpy(returnH, dltH1ToH2, (homography.HomoDim+1)*sizeof(double));
    delete[] dltH1ToH2;
  }

  int RANSACby::countInliers(const vector<HarrisCorner>& corners1, 
    const vector<HarrisCorner>& corners2, const vector<pair<pair<int,int>,double>>& pairs, 
    double* H1ToH2, double marginThres, unsigned int* inliers /*= NULL*/)
  {
    int support = 0;
    for (int i=0; i < (int)pairs.size(); i++)
    {
      CvMat H = cvMat(3,3,CV_64FC1,H1ToH2);
      unsigned int c1 = pairs[i].first.first;
      unsigned int c2 = pairs[i].first.second;
      CvMat* p1 = cvCreateMat(3,1,CV_64FC1);
      cvmSet(p1,0,0,corners1[c1].x);
      cvmSet(p1,1,0,corners1[c1].y);
      cvmSet(p1,2,0,1);
      CvMat* p2 = cvCreateMat(3,1,CV_64FC1);
      cvMatMul(&H,p1,p2);
      double p2x = cvmGet(p2,0,0)/cvmGet(p2,2,0);
      double p2y = cvmGet(p2,1,0)/cvmGet(p2,2,0);
      double dist = sqrt((p2x-corners2[c2].x)*(p2x-corners2[c2].x)+
        (p2y-corners2[c2].y)*(p2y-corners2[c2].y));
      if (dist < marginThres)
      {
        ++support;
        if (inliers != NULL)
          inliers[support-1] = i;
      }
    }
    return support;
  }

  void RANSACby::randomSample(const vector<pair<pair<int,int>,double>>& pairs, unsigned int* randResult, int len)
  {
    //randResult came in with the number pairs.size() at each element. the right number should be 0->len-1
    for (int i=0; i<len; i++)
    {
      randResult[i] = (unsigned int)floor(double(rand())/RAND_MAX*pairs.size());
      for (int j=0; j<i; j++)
      {
        if (randResult[i] == randResult[j] || randResult[i]==pairs.size())
        {
          i--;
          break;
        }
      }
    }
  }

  void RANSACby::randomWeightedSample(const vector<pair<pair<int,int>,double>>& pairs, unsigned int* randResult, int len)
  {
    double* cumulative_partition = new double[pairs.size()];
    memset(cumulative_partition, 0, pairs.size()*sizeof(double));
    cumulative_partition[0] = pairs[0].second;
    for (int i=1; i < (int)pairs.size(); i++)
    {
      cumulative_partition[i] = pairs[i].second;
      cumulative_partition[i] += cumulative_partition[i-1];
    }
    for (int i=0; i < (int)pairs.size(); i++)
    {
      cumulative_partition[i]/=cumulative_partition[pairs.size()-1]; 
    }
    for (int i=0; i<len; i++)
    {
      double randRes = double(rand())/RAND_MAX;
      for (int k=0; k < (int)pairs.size(); k++)
      {
        if (randRes < cumulative_partition[k])
        {
          randResult[i] = k;
          break;
        }
      }
      for (int j=0; j<i; j++)
      {
        if (randResult[i] == randResult[j] || randResult[i]==pairs.size())
        {
          i--;
          break;
        }
      }
    }
  }
}