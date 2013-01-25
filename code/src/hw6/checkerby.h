#ifndef __CHECKERBY_h__
#define __CHECKERBY_h__

#include <cv.h>
#include <vector>
#include "harrisby.h"

using namespace std;

namespace CVisby{

  class Checkerby{
  public:
    Checkerby(IplImage* image=NULL){pImage = image;};
    ~Checkerby(){};
    void SetRowCol(int row, int col){numRow=row;numCol=col;};
    void LabelCorners(vector<HarrisCorner>& cornersRaster, IplImage* image=NULL);
    //draw on pImage.
    void drawCorners(vector<HarrisCorner>& cornersToDraw, int line_type, CvScalar s, double* H=NULL, IplImage* pImageToDraw = NULL);
    void drawCorners(vector<HarrisCorner>& cornersToDraw, int line_type, CvScalar s, double* K, double* Rt, IplImage* pImageToDraw = NULL);

    IplImage* pImage;
    int numRow;
    int numCol;
  private:
    class linePair{
    public:
      linePair(int i, int j, double dist){first = i; second = j; cost = dist;};
      int first;
      int second;
      double cost;
      bool operator < (const linePair& other) {return this->cost < other.cost;};
      bool operator > (const linePair& other) {return this->cost > other.cost;};
    };
    void clusterLines(CvSeq* lines, int k);
    //linesH and linesV are outputs
    void sortLines(CvSeq* lines, CvSeq** linesH, CvSeq** linesV);
    double calcLineMergeCost(double rho_i,double rho_j, double theta_i,double theta_j);
    void fillCorners(CvSeq* linesHor,CvSeq* linesVer, vector<HarrisCorner>& cornersRaster, bool useHarris = true);
  };

}

#endif
