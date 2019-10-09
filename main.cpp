/*
    Examples for using DVR class.
    We use multi-dimensional harmonic oscillator here
    
    **** Eigen3 is required ****
*/
#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include<Eigen/Dense>
#include<iostream>
#include"DVR.h"
using namespace std;
using namespace Eigen;

double potentialFunction(const VectorXd& Coord);

int main()
{
    /* Input example for 2-dimensional*/
    int NDim = 2;
    VectorXi NGrids(2);
    VectorXd mass(2), CoordStart(2), CoordEnd(2);
    NGrids << 50, 50;
    mass << 1.0, 1.0;
    CoordStart << -10.0, -10.0;
    CoordEnd << 10.0, 10.0;
    
    /* Input example for 3-dimensional*/
    // int NDim = 3;
    // VectorXi NGrids(3);
    // VectorXd mass(3), CoordStart(3), CoordEnd(3);
    // double (* PotentialPointer)(const VectorXd& Coord) = potentialFunction;
    // NGrids << 30, 30, 30;
    // mass << 1.0, 1.0, 1.0;
    // CoordStart << -10.0, -10.0, -10.0;
    // CoordEnd << 10.0, 10.0, 10.0;

    /* Input example for 1-dimensional*/
    // int NDim = 1;
    // VectorXi NGrids(1);
    // VectorXd mass(1), CoordStart(1), CoordEnd(1);
    // NGrids << 1000;
    // mass << 1.0;
    // CoordStart << -10.0;
    // CoordEnd << 10.0;

    DVR dvrtest(NDim, NGrids, CoordStart, CoordEnd, mass, potentialFunction);

    cout << "DVR object has been initialized." << endl;

    dvrtest.kernel();

    cout << dvrtest.getEigenValues().block(0, 0, 20, 1) << endl;
    cout << "Program finished normally." << endl;
    return 0;
}


double potentialFunction(const VectorXd& Coord)
{
    double V = 0.0;
    for(int ii = 0; ii < Coord.rows(); ii++)
    {
        V = V + 0.5 * pow(Coord(ii), 2);
    }

    return V;
}