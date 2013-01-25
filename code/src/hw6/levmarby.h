#ifndef __LEVMARBY_h__
#define __LEVMARBY_h__

#include <cv.h>

using namespace std;

namespace CVisby{

  class Levmarby{
  public:
    Levmarby(double* allHomos, double* allRs, double* Ks, double* allXs, double* rodrig, double* disto, int numOb){
      allHs = allHomos;
      allR=allRs; K = Ks; allX = allXs;
      numObs=numOb;
      rodrigR=rodrig;
      distort = disto;
    };
    ~Levmarby(){};
    void lmOpt();
    void Para2Camerapara(double *para);
    void Camerapara2Para(double *para);
    void lmResetH();
  private:
    int numObs;
    double* allHs;
    double* allR;
    double* K;
    double* allX;
    double* rodrigR;
    double* distort;
  };

}

#endif
