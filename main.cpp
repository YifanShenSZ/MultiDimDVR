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

MatrixXd potentialFunction(const VectorXd& Coord, const int& NDim);

int main()
{
    /* Input example for 2-dimensional*/
    int NDim = 2;
    VectorXi NGrids(2);
    VectorXd mass(2), CoordStart(2), CoordEnd(2);
    VectorXd energies;
    MatrixXd states;
    NGrids << 50, 50;
    mass << 1.0, 1.0;
    CoordStart << -5.0, -5.0;
    CoordEnd << 5.0, 5.0;
    
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

    DVR dvrtest(NDim, NGrids, CoordStart, CoordEnd, mass, potentialFunction, 2, true);


    dvrtest.kernel(energies, states, 800, true);
    // cout << energies.rows() << endl;
    cout << energies.block(0,0,20,1) << endl;
    cout << "Program finished normally." << endl;
    return 0;
}


MatrixXd potentialFunction(const VectorXd& Coord, const int& NDim)
{
    // double V = 0.0;
    MatrixXd V(2, 2);
    V = V * 0.0;
    for(int ii = 0; ii < NDim; ii++)
    {
        V(0, 0) = V(0, 0) + 0.5 * pow(Coord(ii), 2);
        V(1, 1) = V(1, 1) + 0.5 * pow(Coord(ii), 2);
    }
    // V(0,1) = 1.0;
    // V(1,0) = 1.0;

    return V;
}