#include"DVR.h"
#include<iostream>
#include<Eigen/Dense>
using namespace Eigen;

/* 
    Construction function with initialization 
*/
DVR::DVR(const int& NDim_, const VectorXi& NGrids_, const VectorXd& CoordStart_, const VectorXd& CoordEnd_, const VectorXd& mass_, double (* PotentialPointer_)(const VectorXd& Coord))
: NDim(NDim_), NGrids(NGrids_), CoordStart(CoordStart_), 
CoordEnd(CoordEnd_), mass(mass_), PotentialPointer(PotentialPointer_)
{
    dx.resize(NDim_);
    for(int ii = 0; ii < NDim_; ii++)
    {
        dx(ii) = (CoordEnd_(ii) - CoordStart_(ii)) / (NGrids_(ii) - 1);
    }
    buildDVR();
}

DVR::~DVR()
{
}

/* 
    Function to return element of one-dimensional DVR kinetic energy matrix.
    It is based on sinc functions. 
*/
double DVR::oneDimK(const double& deltaX, const double& massX, const int& iii, const int& jjj)
{
    double hbar = 1.0, PI = 2 * asin(1.0), kinetics;
    if(iii == jjj) kinetics = pow(hbar, 2) * pow(-1, iii - jjj) / 2.0 / massX / pow(deltaX, 2) * (PI * PI / 3.0);
    else kinetics = pow(hbar, 2) * pow(-1, iii - jjj) / 2.0 / massX / pow(deltaX, 2) * 2.0 / pow(iii - jjj, 2);

    return kinetics;
}


/*
    Initialize and construct Hamiltonian matrix, which is based on (2.6) of the Ref.
    Potential energy is obtained from PotentialPointer.
    Kinetic energy is calculated based on sinc functions.
    For formula details, see the Ref.
*/
void DVR::buildDVR()
{
    int length = 1;
    VectorXd Coord(NDim);
    
    for(int ii = 0; ii < NDim; ii++) 
    {
        length = length * NGrids(ii);
    }

    Hamiltonian.resize(length, length);
    kineticE.resize(length, length);
    potentialE.resize(length, length);
    indice.resize(length, NDim);

    for(int ll = 0; ll < length; ll++)
    {
        int tmpll = ll;
        for(int dd = 0; dd < NDim; dd++)
        {
            int tmp = 1;
            for(int ii = NDim - 1; ii > dd; ii--)
            {
                tmp = tmp * NGrids(ii);
            }
            indice(ll, dd) = tmpll/tmp;
            tmpll = tmpll - indice(ll, dd) * tmp;
        }
    }
    
    /* Calculate potentialE and kineticE matrices */ 
    potentialE = 0.0 * potentialE;
    for(int ii = 0; ii < length; ii++)
    {
        for(int dd = 0; dd < NDim; dd++)
        {
            Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
        }
        potentialE(ii, ii) = PotentialPointer(Coord);
        for(int jj = 0; jj < length; jj++)
        {
            kineticE(ii, jj) = 0.0;
            for(int dd = 0; dd < NDim; dd++)
            {
                bool Ktmp = true;
                for(int ddtmp = 0; ddtmp < NDim; ddtmp++)
                {
                    /* These "if"s are the delta symbols in equ.(2.6) */
                    if(ddtmp != dd)
                    {
                        if(indice(ii,ddtmp) != indice(jj,ddtmp)) Ktmp = false;
                    }
                }
                if(Ktmp) kineticE(ii, jj) = kineticE(ii, jj) + oneDimK(dx(dd), mass(dd), indice(ii, dd), indice(jj, dd));
            }
        }
    }
    
    Hamiltonian = kineticE + potentialE;
}

/*
    Function to calculate eigenvalues and eigenvectors of Hamiltonian
*/
void DVR::kernel()
{
    Eigen::SelfAdjointEigenSolver<MatrixXd> KernelSolver(Hamiltonian);
    eigenValues = KernelSolver.eigenvalues();
    eigenStates = KernelSolver.eigenvectors();
    solved = true;
}

/*
    Public functions to get corresponding private matrices
*/
MatrixXd DVR::getHamiltonian()
{
    return Hamiltonian;
}
MatrixXd DVR::getKineticE()
{
    return kineticE;
}
MatrixXd DVR::getPotentialE()
{
    return potentialE;
}

MatrixXd DVR::getEigenStates()
{
    if(solved) return eigenStates;
    else
    {
        std::cout << "getEigenStates() must be called after kernel()" << std::endl;
        exit(99);
    }
}
VectorXd DVR::getEigenValues()
{
    if(solved) return eigenValues;
    else
    {
        std::cout << "getEigenValues() must be called after kernel()" << std::endl;
        exit(99);
    }
}