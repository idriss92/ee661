#ifndef __UTILIBY_h__
#define __UTILIBY_h__

#include <cv.h>
namespace CVisby{
  
  const double pi = 3.14159265358979323846;

  const int utilibyCoordDim = 3;
  const int utilibyHomoDim = 8;
  const int utilibyRecDim = 4;

  class Utiliby{
  public:
    static void printMatrix(CvMat* mat);
    static void printMatrix(double* m, int row, int col, int mode);
    Utiliby(){worldImage = NULL;}
    IplImage * worldImage;
    void transformImageToWorld(IplImage *pImage, double *H, IplImage **returnImage, int mode, double* inRegion = NULL, double* inClip = NULL);
    void releaseWorldImage(){cvReleaseImage(&worldImage); worldImage = NULL;}
    double getRotateThetaForVerticleLine(CvMat* H, double* p1, double* p2);
    static double convolveMask(double* mask, int radius, IplImage* pImage, int i, int j, bool useNearest = true);
  };

}
#endif
