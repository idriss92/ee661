#ifndef __HOMOGRABY_h__
#define __HOMOGRABY_h__

#include <cv.h>
namespace CVisby{
  class Homograby{
  public:
    Homograby(int k);
    Homograby(int k, int numObs, double *ix, double *iy, double *ixp, double *iyp);
    ~Homograby();
    void importFromFile(const char* fn);
    void importFromUserInput();
    void calcH(double *returnH, int mode); // Assume H is already allocated.
    void calcH_DLT(double *returnH, int mode); // Assume H is already allocated.
    void calcH_BDLT(double *returnH, int mode); // Assume H is already allocated.
    int HomoDim;
  private:
    int CoordDim;
    int RecDim;
    double *x;
    double *y;
    double *xp;
    double *yp;
    //xp,yp = H * x,y;
    double *region; // RecDim*CoordDim, row first.
    //Image coordinates within the rectangle of these four points will appear in the corrected world image.
    int numObs;
    inline double fillLine(const double& x, const double& y, const double &xp,
      const double &yp, const int& j);
    inline void SetDim(int k)
    {
      HomoDim = k*k-1;
      CoordDim = k;
      RecDim = (int)pow(2.0,(double)k-1);
    }
  };
}

#endif
