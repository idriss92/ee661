#ifndef __ZHENGYOUBY_h__
#define __ZHENGYOUBY_h__

#include <cv.h>
namespace CVisby{
  class Zhengyouby{
  public:
    Zhengyouby():nSol(6){};
    void inline initH(double* allH, int numOb){allHs = allH; numObs = numOb;};
    void calcK(double* K);//K is output, as 1*9 vector (K matrix)
    void calcRt(double* Rt, int index, double* rodrig = NULL);//for image No. index, Rt is output, as 1*12 vector (R|t matrix)
    void prepareRt(double* K);
    void calcRadialDistortion(double* K, double* allX, double* distort);//distort[2] is k1 and k2, output. allX is input.
  private:
    void release(){cvReleaseMat(&mKInv);};
    const int nSol;
    double *allHs;
    CvMat* mKInv;
    //Image coordinates within the rectangle of these four points will appear in the corrected world image.
    int numObs;
    inline double fillLine(const int& i, const int& j);
  };
}

#endif
