#include "levmarby.h"
#include "levmar.h"
#include <iostream>
#include <fstream>

namespace CVisby{


// convert the 3-para [vx vy vz] of the Rodigrues
// fomular back into the 3x3 rotation matrix
void V2R(double vx, double vy, double vz, CvMat *R){
  int i;
  // theta = norm(v) because v=theta*w and norm(w)=1
  double theta = sqrt(pow(vx, 2) + pow(vy, 2) + pow(vz, 2));
  double wx = vx/theta, wy = vy/theta, wz = vz/theta;
  double Wdata[9] = {0, -wz, wy,
    wz, 0, -wx,
    -wy, wx, 0};
  CvMat W = cvMat(3, 3, CV_64FC1, Wdata);
  CvMat* Mtmp = cvCreateMat(3, 3, CV_64FC1);
  //R = I + W*sin(theta) + W*W*(1-cos(theta);
  cvZero(R);
  for(i=0; i<3; i++)
    cvmSet(R,i,i,1.0);
  cvmScale(&W,Mtmp,sin(theta));
  cvAdd(R,Mtmp,R);
  cvMatMul(&W,&W,Mtmp);
  cvmScale(Mtmp,Mtmp,1-cos(theta));
  cvAdd(R,Mtmp,R);
  cvReleaseMat(&Mtmp);
}
void CameraCalibrationFunc(double* X, double *para, double tran_X[2], int idx, int isRadial = 1)
{
  double alphax = para[0], alphay = para[1];
  double skew = para[2];
  double x0 = para[3], y0 = para[4];
  double k1 = para[5], k2 = para[6];
  double vx = para[7+6*idx], vy = para[7+6*idx+1], vz = para[7+6*idx+2];
  double tx = para[7+6*idx+3], ty = para[7+6*idx+4], tz = para[7+6*idx+5];
  CvMat* R = cvCreateMat(3, 3, CV_64FC1);
  CvMat* ptX = cvCreateMat(3, 1, CV_64FC1);
  CvMat* ptx = cvCreateMat(3, 1, CV_64FC1);
  double x,y,u,v,value;
  CvMat* K = cvCreateMat(3, 3, CV_64FC1);
  cvZero(K);
  cvmSet(K,2,2,1.0);
  cvmSet(K,0,0,alphax);
  cvmSet(K,0,1,skew);
  cvmSet(K,0,2,x0);
  cvmSet(K,1,1,alphay);
  cvmSet(K,1,2,y0);
  // V convert to R = [r1 r2 r3]
  V2R(vx,vy,vz,R);
  // Set R = [r1,r2,t]
  cvmSet(R,0,2,tx);
  cvmSet(R,1,2,ty);
  cvmSet(R,2,2,tz);
  cvMatMul(K,R,K);
  // Set X
  cvmSet(ptX,0,0,X[0]);
  cvmSet(ptX,1,0,X[1]);
  cvmSet(ptX,2,0,1.0);
  // x = RX u = KRX
  cvMatMul(R,ptX,ptx);
  x = cvmGet(ptx,0,0)/cvmGet(ptx,2,0);
  y = cvmGet(ptx,1,0)/cvmGet(ptx,2,0);
  u = x0 + alphax*x + skew*y;
  v = y0 + alphay*y;
  if(isRadial){
    value = pow(x,2.0)+pow(y,2.0);
    tran_X[0] = u + (u-x0)*(k1*value+k2*pow(value,2.0));
    tran_X[1] = v + (v-y0)*(k1*value+k2*pow(value,2.0));
  }
  else{
    tran_X[0] = u;
    tran_X[1] = v;
  }
  cvReleaseMat(&R);
  cvReleaseMat(&ptX);
  cvReleaseMat(&ptx);
  cvReleaseMat(&K);
}
static void CalculateCameraCalibrationDistFunc(double *para, double *tran_x, int m, int n, void *adata)
{
  int i;
  double* pair;
  pair = (double*)adata;
  for(i=0; i<80*n/160; i++){
    CameraCalibrationFunc(pair+i*4, para, tran_x+i*2, i/80);
  }
}

void Levmarby::Camerapara2Para(double *para)
{
  int i;
  // intrinsic paras
  para[0] = K[0];
  para[1] = K[4];
  para[2] = K[1];
  para[3] = K[2];
  para[4] = K[5];
  // radial distortion paras
  para[5] = distort[0];
  para[6] = distort[1];
  // extrinsic paras
  for(i=0; i<numObs; i++){
    double* RPos =rodrigR+i*6;
    para[7+6*i] = RPos[0];
    para[7+6*i+1] = RPos[1];
    para[7+6*i+2] = RPos[2];
    para[7+6*i+3] = RPos[3];
    para[7+6*i+4] = RPos[4];
    para[7+6*i+5] = RPos[5];
  }
}
void Levmarby::Para2Camerapara(double *para){
  int i,j;
  CvMat *R = cvCreateMat(3,3,CV_64FC1);
  // intrinsic paras
  K[0] = para[0];
  K[4] = para[1];
  K[1] = para[2];
  K[2] = para[3];
  K[5] = para[4];
  // radial distortion paras
  distort[0] = para[5];
  distort[1] = para[6];
  // extrinsic paras
  for(i=0; i<numObs; i++){
    double* RPos =rodrigR+i*6;
    RPos[0] = para[7+6*i];
    RPos[1] = para[7+6*i+1];
    RPos[2] = para[7+6*i+2];
    RPos[3] = para[7+6*i+3];
    RPos[4] = para[7+6*i+4];
    RPos[5] = para[7+6*i+5];
    V2R(RPos[0], RPos[1], RPos[2], R);
    for(j=0; j<3; j++){
      allR[i*12+j*4+0] = cvmGet(R,j,0);
      allR[i*12+j*4+1] = cvmGet(R,j,1);
      allR[i*12+j*4+2] = cvmGet(R,j,2);
    }
    allR[i*12+0*4+3] = para[7+6*i+3];
    allR[i*12+1*4+3] = para[7+6*i+4];
    allR[i*12+2*4+3] = para[7+6*i+5];
  }
  cvReleaseMat(&R);
}

void Levmarby::lmOpt()
{
  int ret;
  double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
  opts[0]=LM_INIT_MU; opts[1]=1E-12; opts[2]=1E-12; opts[3]=1E-15;
  opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used
  void (*err)(double *p, double *hx, int m, int n, void *adata);
  int LM_m = (5+2+numObs*6), LM_n = 2*numObs*80;
  double *ptx = (double *)malloc(LM_n*sizeof(double));
  double *para = (double*)malloc(LM_m*sizeof(double));
  // initialize parameters
  Camerapara2Para(para);
  for(int i=0; i<numObs*80; i++){
    ptx[2*i] = allX[i*4+2];
    ptx[2*i+1] = allX[i*4+3];
  }
  err = CalculateCameraCalibrationDistFunc;
  ret = dlevmar_dif(err, para, ptx, LM_m, LM_n, 1000, opts, info, NULL, NULL,
    allX); // no Jacobian
  ofstream ff;
  ff.open("Info_levmar.txt");
  ff<<"Info[0]:"<<info[0]<<endl;
  ff<<"Info[1]:"<<info[1]<<endl;
  ff<<"Iterations:"<<info[5]<<endl;
  ff.close();
  // store paras
  Para2Camerapara(para);
}

void Levmarby::lmResetH()
{
  for (int i=0;i<numObs;i++)
  {
    double* HPos = allHs+i*9;
    double* RPos = allR+i*12;
    CvMat tempK = cvMat(3,3,CV_64FC1,K);
    CvMat* tempR = cvCreateMat(3,3,CV_64FC1);
    cvmSet(tempR,0,0,RPos[0]);
    cvmSet(tempR,1,0,RPos[4]);
    cvmSet(tempR,2,0,RPos[8]);
    cvmSet(tempR,0,1,RPos[1]);
    cvmSet(tempR,1,1,RPos[5]);
    cvmSet(tempR,2,1,RPos[9]);
    cvmSet(tempR,0,2,RPos[3]);
    cvmSet(tempR,1,2,RPos[7]);
    cvmSet(tempR,2,2,RPos[11]);
    cvMatMul(&tempK,tempR,tempR);
    for (int j1=0; j1<3;j1++)
      for (int j2=0; j2<3;j2++)
        HPos[j1*3+j2]=cvmGet(tempR,j1,j2);
    for (int j1=0; j1<3;j1++)
      for (int j2=0; j2<3;j2++)
        HPos[j1*3+j2]=HPos[j1*3+j2]/HPos[8];
    cvReleaseMat(&tempR);
  }
}

};