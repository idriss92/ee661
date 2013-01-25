#include "zhengyouby.h"
#include "utiliby.h"
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <iostream>
using namespace std;


namespace CVisby{

  void Zhengyouby::calcK(double* K)
  {
    double *A = new double[2*numObs*nSol];//A is column priority
    //double *sol = new double[nSol];
    for (int i=0; i<2*numObs; i++)
    {
      for (int j=0; j<nSol; j++)
      {
        A[i*nSol+j] = fillLine(i,j);
      }
    }
    CvMat MA = cvMat(2*numObs,nSol,CV_64FC1);
    //cvInitMatHeader(&MAT, HomoDim, nb, CV_64FC1, A);
    //cvTranspose(&MAT,pMA);
    cvInitMatHeader(&MA, 2*numObs,nSol,CV_64FC1, A);
    CvMat* U = cvCreateMat(2*numObs,2*numObs,CV_64FC1);
    CvMat* D = cvCreateMat(2*numObs,nSol,CV_64FC1);
    CvMat* V = cvCreateMat(nSol,nSol,CV_64FC1);
    CvMat* VT = cvCreateMat(nSol,nSol,CV_64FC1);
    cvSVD( &MA, D, U ,VT, CV_SVD_U_T|CV_SVD_V_T);
    cvTranspose(VT,V);

    cout<<"A:"<<endl;
    Utiliby::printMatrix(&MA);
    cout<<"V:"<<endl;
    Utiliby::printMatrix(V);
    double B11 = cvmGet(V,0,nSol-1);
    double B12 = cvmGet(V,1,nSol-1);
    double B22 = cvmGet(V,2,nSol-1);
    double B13 = cvmGet(V,3,nSol-1);
    double B23 = cvmGet(V,4,nSol-1);
    double B33 = cvmGet(V,5,nSol-1);
    double v0 = (B12*B13-B11*B23)/(B11*B22-B12*B12);
    double lambda = B33 - (B13*B13 + v0*(B12*B13-B11*B23))/B11;
    double alpha = sqrt(lambda/B11);
    double beta = sqrt(lambda*B11/(B11*B22-B12*B12));
    double gamma = -B12*alpha*alpha*beta/lambda;
    double u0 = gamma*v0/beta-B13*alpha*alpha/lambda;
    K[0]=alpha;
    K[1]=gamma;
    K[2]=u0;
    K[3]=0;
    K[4]=beta;
    K[5]=v0;
    K[6]=0;
    K[7]=0;
    K[8]=1;
    //cvReleaseMat(&&MA);
    cvReleaseMat(&U);
    cvReleaseMat(&D);
    cvReleaseMat(&V);
    cvReleaseMat(&VT);
    delete[] A;
    //delete[] sol;
  }

  void Zhengyouby::calcRt(double* Rt, int index, double* rodrig)
  {
    double* H = allHs+index*9;
    CvMat* mH1 = cvCreateMat(3,1,CV_64FC1);
    CvMat* mH2 = cvCreateMat(3,1,CV_64FC1);
    CvMat* mH3 = cvCreateMat(3,1,CV_64FC1);
    cvmSet(mH1,0,0,H[0]);
    cvmSet(mH1,1,0,H[3]);
    cvmSet(mH1,2,0,H[6]);
    cvmSet(mH2,0,0,H[1]);
    cvmSet(mH2,1,0,H[4]);
    cvmSet(mH2,2,0,H[7]);
    cvmSet(mH3,0,0,H[2]);
    cvmSet(mH3,1,0,H[5]);
    cvmSet(mH3,2,0,H[8]);
    //cout << "H:"<< endl;
    //Utiliby::printMatrix(H,3,3,0);
    CvMat* KInvH1 = cvCreateMat(3,1,CV_64FC1);
    CvMat* KInvH2 = cvCreateMat(3,1,CV_64FC1);
    CvMat* KInvH3 = cvCreateMat(3,1,CV_64FC1);
    //cout << "KInv:"<< endl;
    //Utiliby::printMatrix(mKInv);
    cvMatMul(mKInv,mH1,KInvH1);
    cvMatMul(mKInv,mH2,KInvH2);
    cvMatMul(mKInv,mH3,KInvH3);
    CvMat* r1 = cvCreateMat(3,1,CV_64FC1);
    CvMat* r2 = cvCreateMat(3,1,CV_64FC1);
    CvMat* r3 = cvCreateMat(3,1,CV_64FC1);
    CvMat* t = cvCreateMat(3,1,CV_64FC1);
    double lambdaForT = 1/sqrt(cvDotProduct(KInvH1,KInvH1));
    //cvNormalize(KInvH1,r1);
    //cvNormalize(KInvH2,r2);
    //r1=cvCloneMat(KInvH1);
    //r2=cvCloneMat(KInvH2);
    //t=cvCloneMat(KInvH3);
    cvmSet(r1,0,0,cvmGet(KInvH1,0,0)*lambdaForT);
    cvmSet(r1,1,0,cvmGet(KInvH1,1,0)*lambdaForT);
    cvmSet(r1,2,0,cvmGet(KInvH1,2,0)*lambdaForT);
    cvmSet(r2,0,0,cvmGet(KInvH2,0,0)*lambdaForT);
    cvmSet(r2,1,0,cvmGet(KInvH2,1,0)*lambdaForT);
    cvmSet(r2,2,0,cvmGet(KInvH2,2,0)*lambdaForT);
    cvCrossProduct(r1,r2,r3);
    cvmSet(t,0,0,cvmGet(KInvH3,0,0)*lambdaForT);
    cvmSet(t,1,0,cvmGet(KInvH3,1,0)*lambdaForT);
    cvmSet(t,2,0,cvmGet(KInvH3,2,0)*lambdaForT);

    //Take a second to verify K*[r1|r2|t]=H
    /*CvMat* tempMat = cvCreateMat(3,3,CV_64FC1);
    CvMat* tempMat2 = cvCreateMat(3,3,CV_64FC1);
    CvMat tempMatH = cvMat(3,3,CV_64FC1,H);
    cvMatMul(mKInv,&tempMatH,tempMat2);
    cvmSet(tempMat,0,0,cvmGet(r1,0,0));
    cvmSet(tempMat,1,0,cvmGet(r1,1,0));
    cvmSet(tempMat,2,0,cvmGet(r1,2,0));
    cvmSet(tempMat,0,1,cvmGet(r2,0,0));
    cvmSet(tempMat,1,1,cvmGet(r2,1,0));
    cvmSet(tempMat,2,1,cvmGet(r2,2,0));
    cvmSet(tempMat,0,2,cvmGet(t,0,0));
    cvmSet(tempMat,1,2,cvmGet(t,1,0));
    cvmSet(tempMat,2,2,cvmGet(t,2,0));

    cout << "r1r2t:" << endl;
    Utiliby::printMatrix(tempMat);
    cout << "Kinv*H:" << endl;
    Utiliby::printMatrix(tempMat2);*/

    CvMat *Q = cvCreateMat(3,3,CV_64FC1);
    CvMat *QU = cvCreateMat(3,3,CV_64FC1);
    CvMat *transQU = cvCreateMat(3,3,CV_64FC1);
    CvMat *QD = cvCreateMat(3,3,CV_64FC1);
    CvMat *QV = cvCreateMat(3,3,CV_64FC1);
    CvMat *R = cvCreateMat(3,3,CV_64FC1);
    for(int j=0; j<3; j++){
      cvmSet(Q,j,0,cvmGet(r1,j,0));
      cvmSet(Q,j,1,cvmGet(r2,j,0));
      cvmSet(Q,j,2,cvmGet(r3,j,0));
    }
    cvSVD(Q, QD, QU, QV, CV_SVD_U_T|CV_SVD_V_T);
    cvTranspose(QU, transQU);
    cvMatMul(transQU, QV, R);
    if (rodrig != NULL)
    {
      double trace = (cvTrace(R)).val[0]; // cvTrace returns a CvScalar
      double theta = acos((trace - 1) / 2.0);
      double value1 = theta/(2*sin(theta));
      rodrig[0] = value1*(cvmGet(R, 2, 1)-cvmGet(R, 1, 2));
      rodrig[1] = value1*(cvmGet(R, 0, 2)-cvmGet(R, 2, 0));
      rodrig[2] = value1*(cvmGet(R, 1, 0)-cvmGet(R, 0, 1));
      rodrig[3] = cvmGet(t,0,0);
      rodrig[4] = cvmGet(t,1,0);
      rodrig[5] = cvmGet(t,2,0);
    }
    for (int i=0; i<3;i++)
    {
      for(int j=0; j<3;j++)
      {
        Rt[i*4+j]=cvmGet(R,i,j);
      }
      Rt[i*4+3]=cvmGet(t,i,0);
    }
    cvReleaseMat(&mH1);
    cvReleaseMat(&mH2);
    cvReleaseMat(&mH3);
    cvReleaseMat(&KInvH1);
    cvReleaseMat(&KInvH2);
    cvReleaseMat(&KInvH3);
    cvReleaseMat(&r1);
    cvReleaseMat(&r2);
    cvReleaseMat(&r3);
    cvReleaseMat(&t);
    cvReleaseMat(&Q);
    cvReleaseMat(&QU);
    cvReleaseMat(&transQU);
    cvReleaseMat(&QD);
    cvReleaseMat(&QV);
    cvReleaseMat(&R);
  }

  void Zhengyouby::prepareRt(double *K)
  {
    CvMat mK = cvMat(3,3,CV_64FC1);
    cvInitMatHeader(&mK,3,3,CV_64FC1,K);
    mKInv = cvCreateMat(3,3,CV_64FC1);
    cvInvert(&mK,mKInv,CV_LU);
  }

  inline double Zhengyouby::fillLine(const int& i, const int& j)
  {
    double* H = allHs+(i/2)*9;
    if (i%2==0)
    {
      switch (j)
      {
      case 0:
        return H[0]*H[1];
      case 1:
        return H[0]*H[4]+H[3]*H[1];
      case 2:
        return H[3]*H[4];
      case 3:
        return H[6]*H[1]+H[0]*H[7];
      case 4:
        return H[6]*H[4]+H[3]*H[7];
      case 5:
        return H[6]*H[7];
      default:
        return 0;
      }
    }
    else
    {
      switch (j)
      {
      case 0:
        return H[0]*H[0]-H[1]*H[1];
      case 1:
        return 2*H[0]*H[3]-2*H[1]*H[4];
      case 2:
        return H[3]*H[3]-H[4]*H[4];
      case 3:
        return 2*H[0]*H[6]-2*H[1]*H[7];
      case 4:
        return 2*H[3]*H[6]-2*H[4]*H[7];
      case 5:
        return H[6]*H[6] - H[7]*H[7];
      default:
        return 0;
      }
    }
  }

  void Zhengyouby::calcRadialDistortion(double* K, double* allX, double* distort)
  {
    double alphax = K[0];
    double alphay = K[4];
    double skew = K[1];
    double x0 = K[2];
    double y0 = K[5];
    double u,v,x,y;
    double value1, value2, value3;
    CvMat *R = cvCreateMat(3,3,CV_64FC1);
    CvMat *ptX = cvCreateMat(3,1,CV_64FC1);
    CvMat *ptx = cvCreateMat(3,1,CV_64FC1);
    CvMat *D = cvCreateMat(2*80*numObs,2,CV_64FC1);
    CvMat *d = cvCreateMat(2*80*numObs,1,CV_64FC1);
    CvMat *solk = cvCreateMat(2,1,CV_64FC1);
    CvMat *DT = cvCreateMat(2,2*80*numObs,CV_64FC1);
    CvMat *tmp = cvCreateMat(2,2,CV_64FC1);
    int idx = 0;
    for(int i=0; i<numObs; i++){
      // set H
      for(int j=0; j<3; j++)
        for(int k=0; k<3; k++)
          cvmSet(R,j,k,allHs[i*9+3*j+k]);
      // for each point correspondence
      for(int j=0; j<80; j++){
        cvmSet(ptX,0,0,allX[i*320+j*4+0]);
        cvmSet(ptX,1,0,allX[i*320+j*4+1]);
        cvmSet(ptX,2,0,1.0);
        cvMatMul(R,ptX,ptx);
        u = cvmGet(ptx,0,0)/cvmGet(ptx,2,0);
        v = cvmGet(ptx,1,0)/cvmGet(ptx,2,0);
        //printf("(%f %f), (%f %f)\n",
        //u,v,ptspairs.inlierp2[idx].x,ptspairs.inlierp2[idx].y);
        x = (u-x0)/alphax;
        y = (v-y0)/alphay;
        value1 = u - x0;
        value2 = v - y0;
        value3 = pow(x,2.0)+pow(y,2.0);
        // add two equations
        cvmSet(D,2*idx, 0,value1*value3);
        cvmSet(D,2*idx, 1,value1*pow(value3,2.0));
        cvmSet(D,2*idx+1,0,value2*value3);
        cvmSet(D,2*idx+1,1,value2*pow(value3,2.0));
        cvmSet(d,2*idx, 0, allX[i*320+j*4+2]-u);
        cvmSet(d,2*idx+1,0, allX[i*320+j*4+3]-v);
        idx++;
      }
    }
    // solve k
    // k = (D^T * D)^-1 *D^T * d
    cvSolve(D,d,solk,CV_SVD);
    distort[0] = cvmGet(solk,0,0);
    distort[1] = cvmGet(solk,1,0);
    // Release
    cvReleaseMat(&R);
    cvReleaseMat(&ptX);
    cvReleaseMat(&ptx);
    cvReleaseMat(&D);
    cvReleaseMat(&d);
    cvReleaseMat(&solk);
  }

}
