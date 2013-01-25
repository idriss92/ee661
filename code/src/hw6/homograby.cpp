#include "homograby.h"
#include "utiliby.h"
#include <cv.h>
#include <highgui.h>
#include <math.h>

namespace CVisby{

  Homograby::Homograby(int k)
  {
    SetDim(k);
    x = y = xp = yp = NULL;
    numObs = 0;
    region = new double[RecDim*CoordDim];
    for (int r=0; r<RecDim; r++)
    {
      for (int i=0; i<CoordDim-1;i++)
      {
        region[i*RecDim+r]=-1;
      }
      region[(CoordDim-1)*RecDim+r]=1;
    }
  }

  Homograby::~Homograby()
  {
    if ( x != NULL ) delete[] x;
    if ( y != NULL ) delete[] y;
    if ( xp != NULL ) delete[] xp;
    if ( yp != NULL ) delete[] yp;
    delete[] region;
  }

  Homograby::Homograby(int k, int numObs, double *ix, double *iy, double *ixp, double *iyp): numObs(numObs)
  {
    SetDim(k);
    x = new double[numObs];
    y = new double[numObs];
    xp = new double[numObs];
    yp = new double[numObs];
    memcpy(x, ix, numObs * sizeof(double));
    memcpy(y, iy, numObs * sizeof(double));
    memcpy(xp, ixp, numObs * sizeof(double));
    memcpy(yp, iyp, numObs * sizeof(double));
    region = new double[CoordDim*RecDim];
    for (int r=0; r<RecDim; r++)
    {
      for (int i=0; i<CoordDim-1;i++)
      {
        region[i*RecDim+r]=-1;
      }
      region[(CoordDim-1)*RecDim+r]=1;
    }
  }

  void Homograby::importFromFile(const char* fn)
  {
    FILE *fp = fopen(fn, "r");
    CV_Assert(fp);
    fscanf(fp, "%d\n", &numObs);
    if ( x != NULL ) delete[] x;
    if ( y != NULL ) delete[] y;
    if ( xp != NULL ) delete[] xp;
    if ( yp != NULL ) delete[] xp;
    x = new double[numObs];
    y = new double[numObs];
    xp = new double[numObs];
    yp = new double[numObs];
    for (int i=0; i < numObs; i++)
    {
      fscanf(fp, "%lf, %lf, %lf, %lf\n", &x[i], &y[i], &xp[i], &yp[i]);
    }
    char isHomoRangeDefined[30];
    fscanf(fp,"%s",isHomoRangeDefined);
    if (!strcmp(isHomoRangeDefined, "Range"))
    {
      for (int r=0; r<RecDim; r++)
      {
        fscanf(fp,"%lf,%lf\n",&region[0*RecDim+r],&region[1*RecDim+r]);
      }
    }
    fclose(fp);
  }

  void Homograby::importFromUserInput()
  {
    printf("Please input the number of points you prepare to input:\n");
    scanf("%d", &numObs);
    if ( x != NULL ) delete[] x;
    if ( y != NULL ) delete[] y;
    if ( xp != NULL ) delete[] xp;
    if ( yp != NULL ) delete[] xp;
    x = new double[numObs];
    y = new double[numObs];
    xp = new double[numObs];
    yp = new double[numObs];

    for (int i=0; i < numObs; i++)
    {
      printf("Please input the coordinates in the sequence: x_image, y_image, x_world, y_world for point %d:\n", i+1);
      scanf("%lf",&x[i]);
      scanf("%lf",&y[i]);
      scanf("%lf",&xp[i]);
      scanf("%lf",&yp[i]);
    }
    printf("Do you like to specify four points for which the image within it will be corrected?\n If you don't specify, the entire image will be corrected.\n(y/n)?");
    char ans;
    getchar();
    ans = getchar();
    if (ans == 'y')
    {
      for (int r=0; r<RecDim; r++)
      {
        printf("Input a point on image you would like to use for the range:\nx_image, y_image\n");
        scanf("%lf",&region[0*RecDim+r]);
        scanf("%lf",&region[1*RecDim+r]);
      }
    }
  }

  void Homograby::calcH(double *returnH, int mode)//This function can be adapted for any dimension
  {
    int lowDim = CoordDim;
    int nb = (lowDim-1)*numObs;
    double *A = new double[nb*HomoDim];//A is column priority
    double *b = new double[nb];
    double *sol = new double[nb];
    for (int i=0; i<nb; i++)
    {
      for (int j=0; j<HomoDim; j++)
      {
        //A[j*nb+i] = fillLine( x[i/(lowDim-1)], y[i/(lowDim-1)], xp[i/(lowDim-1)], yp[i/(lowDim-1)], (i%(lowDim-1))*HomoDim + j );
        A[i*HomoDim+j] = fillLine( x[i/(lowDim-1)], y[i/(lowDim-1)], xp[i/(lowDim-1)], yp[i/(lowDim-1)], (i%(lowDim-1))*HomoDim + j );
      }
      b[i] = (i%(lowDim-1) == 0)? x[i/(lowDim-1)] : y[i/(lowDim-1)];
    }
    //If use LAPACK
    //int info = 0;
    //dgels_('N', &lb, &HomoDim, &lb, A, &lb, b, &lb, sol, &lb, &info);
    //Else, still choose to go with OpenCV
    //CvMat MAT;
    CvMat *pMA = cvCreateMat(nb,HomoDim,CV_64FC1);
    CvMat Mb;
    CvMat *pMsol = cvCreateMat(HomoDim, 1, CV_64FC1);
    //cvInitMatHeader(&MAT, HomoDim, nb, CV_64FC1, A);
    //cvTranspose(&MAT,pMA);
    cvInitMatHeader(pMA, nb,HomoDim,CV_64FC1, A);
    cvInitMatHeader(&Mb,nb,1, CV_64FC1, b);
    cvSolve(pMA, &Mb, pMsol, CV_LU);

    for (int i=0; i<lowDim; i++)
      for (int j=0; j<lowDim; j++)
      {
        if (i*lowDim+j != HomoDim) returnH[i*lowDim+j] = cvmGet(pMsol, i*lowDim+j, 0);
        else returnH[HomoDim] = 1; 
      }
      //returnH is row priorit
    if (mode == 1)// mode == 0: returnH is from world to image, mode == 1: returnH is from image to world
    {
      CvMat mH;
      cvInitMatHeader(&mH, CoordDim, CoordDim, CV_64FC1, returnH);
      CvMat *pMInvH = cvCreateMat(CoordDim, CoordDim, CV_64FC1);
      cvInvert(&mH, pMInvH, CV_LU);
      for (int i=0; i<lowDim; i++)
        for (int j=0; j<lowDim; j++)
        {
          returnH[i*lowDim+j] = cvmGet(pMInvH, i, j);
        }
    }
    cvReleaseMat(&pMA);
    cvReleaseMat(&pMsol);
    delete[] A;
    delete[] b;
    delete[] sol;
  }

  void Homograby::calcH_BDLT(double *returnH, int mode)//This function can be adapted for any dimension
  {
    int lowDim = CoordDim;
    int nb = (lowDim-1)*numObs;
    double *A = new double[nb*(HomoDim+1)];//A is column priority
    //double *b = new double[nb];
    //double *sol = new double[nb];
    for (int i=0; i<nb; i++)
    {
      for (int j=0; j<HomoDim; j++)
      {
        //A[j*nb+i] = fillLine( x[i/(lowDim-1)], y[i/(lowDim-1)], xp[i/(lowDim-1)], yp[i/(lowDim-1)], (i%(lowDim-1))*HomoDim + j );
        A[i*(HomoDim+1)+j] = fillLine( x[i/(lowDim-1)], y[i/(lowDim-1)], xp[i/(lowDim-1)], yp[i/(lowDim-1)], (i%(lowDim-1))*HomoDim + j );
      }
      A[i*(HomoDim+1)+HomoDim] = (i%(lowDim-1) == 0)? -x[i/(lowDim-1)] : -y[i/(lowDim-1)];
    }
    //If use LAPACK
    //int info = 0;
    //dgels_('N', &lb, &HomoDim, &lb, A, &lb, b, &lb, sol, &lb, &info);
    //Else, still choose to go with OpenCV
    //CvMat MAT;
    CvMat pMA = cvMat(nb,HomoDim+1,CV_64FC1,A);
    CvMat* UT = cvCreateMat(nb,nb,CV_64FC1);
    CvMat* VT = cvCreateMat(HomoDim+1,HomoDim+1,CV_64FC1);
    CvMat* V = cvCreateMat(HomoDim+1,HomoDim+1,CV_64FC1);
    CvMat* W = cvCreateMat(nb,HomoDim+1,CV_64FC1);
    cvSVD(&pMA,W,UT,VT,CV_SVD_U_T|CV_SVD_V_T);
    cvTranspose(VT,V);


    //Utiliby::printMatrix(&pMA);
    for (int i=0; i<lowDim; i++)
      for (int j=0; j<lowDim; j++)
        returnH[i*lowDim+j] = cvmGet(V, i*lowDim+j, HomoDim);

    //returnH is row priority
    if (mode == 1)// mode == 0: returnH is from world to image, mode == 1: returnH is from image to world
    {
      CvMat mH;
      cvInitMatHeader(&mH, CoordDim, CoordDim, CV_64FC1, returnH);
      CvMat *pMInvH = cvCreateMat(CoordDim, CoordDim, CV_64FC1);
      cvInvert(&mH, pMInvH, CV_LU);
      for (int i=0; i<lowDim; i++)
        for (int j=0; j<lowDim; j++)
        {
          returnH[i*lowDim+j] = cvmGet(pMInvH, i, j);
        }
        cvReleaseMat(&pMInvH);
    }

    if (fabs(returnH[HomoDim]) > 0.000000001 )
    {
      for (int i=0; i<lowDim; i++)
        for (int j=0; j<lowDim; j++)
          returnH[i*lowDim+j] /= returnH[HomoDim];
    }

    //cvReleaseMat(&&pMA);
    cvReleaseMat(&W);
    cvReleaseMat(&UT);
    cvReleaseMat(&VT);
    cvReleaseMat(&V);
    delete[] A;
  }

  void Homograby::calcH_DLT(double *returnH, int mode)//This function can be adapted for any dimension
  {
    //normalizing the coordinates
    double cenX = 0, cenY = 0, cenXP = 0, cenYP = 0;
    for (int i=0; i<numObs; i++)
    {
      cenX+=x[i];
      cenXP+=xp[i];
      cenY+=y[i];
      cenYP+=yp[i];
    }
    cenX/=numObs;
    cenY/=numObs;
    cenXP/=numObs;
    cenYP/=numObs;

    double dist = 0;
    double distP = 0;
    for (int i=0; i<numObs; i++)
    {
      dist += sqrt((x[i]-cenX)*(x[i]-cenX)+(y[i]-cenY)*(y[i]-cenY));
      distP += sqrt((xp[i]-cenXP)*(xp[i]-cenXP)+(yp[i]-cenYP)*(yp[i]-cenYP));
    }
    dist /= numObs;
    distP /= numObs;

    dist /= sqrt(2.0);
    distP /= sqrt(2.0);
    
    CvMat* T = cvCreateMat(3,3,CV_64FC1);
    CvMat* TP = cvCreateMat(3,3,CV_64FC1);
    cvZero(T);
    cvZero(TP);
    cvmSet(T,0,0,1.0/dist);
    cvmSet(T,1,1,1.0/dist);
    cvmSet(T,2,2,1.0);
    cvmSet(T,0,2,-1.0/dist*cenX);
    cvmSet(T,1,2,-1.0/dist*cenY);
    cvmSet(TP,0,0,1.0/distP);
    cvmSet(TP,1,1,1.0/distP);
    cvmSet(TP,2,2,1.0);
    cvmSet(TP,0,2,-1.0/distP*cenXP);
    cvmSet(TP,1,2,-1.0/distP*cenYP);

    for (int i=0; i < numObs; i++)
    {
      CvMat* p1 = cvCreateMat(3,1,CV_64FC1);
      CvMat* p2 = cvCreateMat(3,1,CV_64FC1);
      cvmSet(p1,0,0,x[i]);
      cvmSet(p1,1,0,y[i]);
      cvmSet(p1,2,0,1);
      cvmSet(p2,0,0,xp[i]);
      cvmSet(p2,1,0,yp[i]);
      cvmSet(p2,2,0,1);
      CvMat* p1t = cvCreateMat(3,1,CV_64FC1);
      CvMat* p2t = cvCreateMat(3,1,CV_64FC1);
      cvMatMul(T,p1,p1t);
      cvMatMul(TP,p2,p2t);
      x[i] = cvmGet(p1t,0,0)/cvmGet(p1t,2,0);
      y[i] = cvmGet(p1t,1,0)/cvmGet(p1t,2,0);
      xp[i] = cvmGet(p2t,0,0)/cvmGet(p2t,2,0);
      yp[i] = cvmGet(p2t,1,0)/cvmGet(p2t,2,0);
      cvReleaseMat(&p1);
      cvReleaseMat(&p1t);
      cvReleaseMat(&p2);
      cvReleaseMat(&p2t);
    }

    int lowDim = CoordDim;
    int nb = (lowDim-1)*numObs;
    double *A = new double[nb*(HomoDim+1)];//A is column priority
    //double *b = new double[nb];
    //double *sol = new double[nb];
    for (int i=0; i<nb; i++)
    {
      for (int j=0; j<HomoDim; j++)
      {
        //A[j*nb+i] = fillLine( x[i/(lowDim-1)], y[i/(lowDim-1)], xp[i/(lowDim-1)], yp[i/(lowDim-1)], (i%(lowDim-1))*HomoDim + j );
        A[i*(HomoDim+1)+j] = fillLine( x[i/(lowDim-1)], y[i/(lowDim-1)], xp[i/(lowDim-1)], yp[i/(lowDim-1)], (i%(lowDim-1))*HomoDim + j );
      }
      A[i*(HomoDim+1)+HomoDim] = (i%(lowDim-1) == 0)? -x[i/(lowDim-1)] : -y[i/(lowDim-1)];
    }
    //If use LAPACK
    //int info = 0;
    //dgels_('N', &lb, &HomoDim, &lb, A, &lb, b, &lb, sol, &lb, &info);
    //Else, still choose to go with OpenCV
    //CvMat MAT;
    CvMat pMA = cvMat(nb,HomoDim+1,CV_64FC1,A);
    CvMat* UT = cvCreateMat(nb,nb,CV_64FC1);
    CvMat* VT = cvCreateMat(HomoDim+1,HomoDim+1,CV_64FC1);
    CvMat* V = cvCreateMat(HomoDim+1,HomoDim+1,CV_64FC1);
    CvMat* W = cvCreateMat(nb,HomoDim+1,CV_64FC1);
    cvSVD(&pMA,W,UT,VT,CV_SVD_U_T|CV_SVD_V_T);
    cvTranspose(VT,V);

    for (int i=0; i<lowDim; i++)
      for (int j=0; j<lowDim; j++)
      {
        returnH[i*lowDim+j] = cvmGet(V, i*lowDim+j, HomoDim);
      }

    CvMat* H2ToH1 = cvCreateMat(3,3,CV_64FC1);
    for (int i=0; i<lowDim; i++)
      for (int j=0; j<lowDim; j++)
      {
        cvmSet(H2ToH1,i,j,returnH[i*lowDim+j]);
      }
    CvMat* TInv = cvCreateMat(3,3,CV_64FC1);
    cvInvert(T,TInv,CV_SVD);
    cvMatMul(TInv,H2ToH1,H2ToH1);
    cvMatMul(H2ToH1,TP,H2ToH1);
    for (int i=0; i<lowDim; i++)
      for (int j=0; j<lowDim; j++)
      {
        returnH[i*lowDim+j]=cvmGet(H2ToH1,i,j);
      }
    
    cvReleaseMat(&H2ToH1);
    cvReleaseMat(&T);
    cvReleaseMat(&TP);
    cvReleaseMat(&TInv);

    //returnH is row priority
    if (mode == 1)// mode == 0: returnH is from world to image, mode == 1: returnH is from image to world
    {
      CvMat mH;
      cvInitMatHeader(&mH, CoordDim, CoordDim, CV_64FC1, returnH);
      CvMat *pMInvH = cvCreateMat(CoordDim, CoordDim, CV_64FC1);
      cvInvert(&mH, pMInvH, CV_LU);
      for (int i=0; i<lowDim; i++)
        for (int j=0; j<lowDim; j++)
        {
          returnH[i*lowDim+j] = cvmGet(pMInvH, i, j);
        }
      cvReleaseMat(&pMInvH);
    }
  
    if (fabs(returnH[HomoDim]) > 0.000000001 )
    {
      for (int i=0; i<lowDim; i++)
        for (int j=0; j<lowDim; j++)
          returnH[i*lowDim+j] /= returnH[HomoDim];
    }
    Utiliby::printMatrix(returnH,3,3,0);
    //cvReleaseMat(&&pMA);
    cvReleaseMat(&W);
    cvReleaseMat(&UT);
    cvReleaseMat(&VT);
    cvReleaseMat(&V);
    delete[] A;
  }

  inline double Homograby::fillLine(const double& x, const double& y, const double &xp, const double &yp, const int& j)//This function is only coded for 2D case.
  {
    switch (j)
    {
    case 0:
    case 11:
      return xp;
    case 1:
    case 12:
      return yp;
    case 2:
    case 13:
      return 1;
    case 3:
    case 4:
    case 5:
    case 8:
    case 9:
    case 10:
      return 0;
    case 6:
      return -x*xp;
    case 7:
      return -x*yp;
    case 14:
      return -y*xp;
    case 15:
      return -y*yp;
    default:
      return 0;
    }
  }
}
