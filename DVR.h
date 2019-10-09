#ifndef DVR_H_
#define DVR_H_
#include<Eigen/Dense>
using namespace Eigen;

/*
    Class for multidimensional cartesian coordinates DVR solver
    Ref: J. Chem. Phys. 96, 1982 (1992); https://doi.org/10.1063/1.462100
    
    **** Eigen3 is required ****

    Parameters:
        NDim:               Dimension number
        CoordStart:         Starting points for different dimension
        CoordEnd:           Ending points for different dimension
        mass:               Effective mass for different dimension
        dx:                 Delta x for different dimension
        potentialE:         Potential energy matrix
        kineticE:           Kinetic energy matrix
        Hamiltonian:        Hamiltonian
        NGrids:             Number of grids for different dimension
        indice:             A special matrix to note indice in N-dimension to final H matrix
        solved:             A bool to note whether eigen-states have been solved
        PotentialPointer:   Function pointer to input potential function
        
    An example of this class can be found in main.cpp
*/
class DVR
{
private:
    int NDim;
    VectorXd CoordStart, CoordEnd, mass, dx, eigenValues;
    MatrixXd potentialE, kineticE, Hamiltonian, eigenStates;
    MatrixXi NGrids, indice;
    bool solved = false;
    double (* PotentialPointer)(const VectorXd& Coord);

    /* Initialization */
    void buildDVR();
public:
    /* Construction function */
    DVR(const int& NDim_, const VectorXi& NGrids_, const VectorXd& CoordStart_, const VectorXd& CoordEnd_, const VectorXd& mass_, double (* PotentialPointer_)(const VectorXd& Coord));
	
    ~DVR();

    /* Function to return element of one-dimensional DVR kinetic energy matrix based on sinc */
    double oneDimK(const double& deltaX, const double& massX, const int& iii, const int& jjj);
    
    /* Solving the eigen-values and eigen-vectors for Hamiltonian */
    void kernel();

    /* Public functions to get corresponding private matrices */
    MatrixXd getHamiltonian();
    MatrixXd getKineticE();
    MatrixXd getPotentialE();
    MatrixXd getEigenStates();
    VectorXd getEigenValues();
};

#endif
