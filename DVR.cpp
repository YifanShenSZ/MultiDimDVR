#include"DVR.h"
#include<iostream>
#include<Eigen/Dense>
#include"mkl_interface.h"
#include<omp.h>
using namespace Eigen;
using namespace std;

/* 
    Construction function with initialization 
*/
DVR::DVR(const int& NDim_, const VectorXi& NGrids_, const VectorXd& CoordStart_, const VectorXd& CoordEnd_, const VectorXd& mass_, double (* PotentialPointer_)(const VectorXd& Coord, const int& ND), const bool &saveMem_)
: NDim(NDim_), NGrids(NGrids_), CoordStart(CoordStart_), saveMem(saveMem_), 
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
    double kinetics;
    if(iii == jjj) kinetics = 1.0 / 2.0 / massX / pow(deltaX, 2) * (PI * PI / 3.0);
    else kinetics = pow(-1, (iii - jjj)%2 ) / 2.0 / massX / pow(deltaX, 2) * 2.0 / pow(iii - jjj, 2);

    return kinetics;
}


/*
    Initialize and construct Hamiltonian matrix, which is based on (2.6) of the Ref.
    Potential energy is obtained from PotentialPointer.
    Kinetic energy is calculated based on sinc functions.
    For formula details, see the Ref.

    saveMem is a special bool. If it is true, only length will be calculated. None matrices
    are initialized to save memory. And in kernel(), some special functions will be used in
    Lanczos Algorithm.  
*/
void DVR::buildDVR()
{
    length = 1;
    VectorXd Coord(NDim);
    
    for(int ii = 0; ii < NDim; ii++) 
    {
        length = length * NGrids(ii);
    }

    /* Calculate indices */
    indice.resize(length, NDim);
    #pragma omp parallel for
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

    if(saveMem)
    {
        cout << "saveMem mode is on." << endl; 
        /* 
            When saveMem mode is on. Hamiltonian will not be constructed explicitly. 
            Instead, it will be resized to an (NDim+1)-by-length matrix.
            The first row stores the potential energy.
            The others store the kinetic energy for different dimension using the delta iii - jjj.
        */
        Hamiltonian.resize(NDim + 1,length);
        for(int ii = 0; ii < length; ii++)
        {
            for(int dd = 0; dd < NDim; dd++)
            {
                Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
                Hamiltonian(dd + 1, ii) = oneDimK(dx(dd), mass(dd), 0, ii);
            }
            Hamiltonian(0, ii) = PotentialPointer(Coord, NDim);
        }
    }
    else
    {
        Hamiltonian.resize(length, length);

        /* Calculate potentialE and kineticE matrices, add them together */ 
        Hamiltonian = 0.0 * Hamiltonian;
        for(int ii = 0; ii < length; ii++)
        {
            for(int dd = 0; dd < NDim; dd++)
            {
                Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
            }
            Hamiltonian(ii, ii) += PotentialPointer(Coord, NDim);
            // std::cout << Coord(0) << "\t" << potentialE(ii, ii) << std::endl;
            for(int jj = 0; jj < length; jj++)
            {
                double tmp = 0.0;
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
                    if(Ktmp) tmp = tmp + oneDimK(dx(dd), mass(dd), indice(ii, dd), indice(jj, dd));
                }
                Hamiltonian(ii, jj) += tmp;
            }
        }
    }
    cout << "DVR object has been initialized." << endl << endl;
}

/*
    Function to calculate eigenvalues and eigenvectors of Hamiltonian
    solverSize is how many lowest energies calculated.

    If it is 0, solve all eigenvalues and eigenvectors using Eigen::SelfAdjointEigenSolver. 
    If it is k != 0, solve lowest k eigenvalues and eigenvectors.
        In this case, when saveMem is used, perform a reorthogonalized Lanczos Algorithm.
        Otherwise, use intel mkl interface to perform LAPACKE_dsyevr.

    lanzcos is a bool to determine whether to use (reorthogonalized) Lanczos Algorithm
*/
void DVR::kernel(VectorXd& energies, MatrixXd& states)
{
    if(saveMem)
    {
        cout << endl << "When saveMem is on, one can only use kernel() with solverSize." << endl;
    }
    SelfAdjointEigenSolver<MatrixXd> KernelSolver(Hamiltonian);
    eigenValues = KernelSolver.eigenvalues();
    eigenStates = KernelSolver.eigenvectors();
    energies = eigenValues;
    states = eigenStates;
}
void DVR::kernel(VectorXd& energies, MatrixXd& states, const int& solverSize, const bool& lanczos)
{
    if(solverSize <= 0)
    {
        cout << "Input a wrong solverSize. It must be positive integer from 1 to length." << endl;
        exit(99);
    }
    else
    {
        cout << solverSize << " eigenvalues and eigenvectors will be evaluated." << endl;
        if(saveMem)
        {
            if(!lanczos) cout << endl << "saveMem mode is on. Automatically switch to Lanczos Algorithm." << solverSize << endl;
            cout << "The reorthogonalized Lanczos Algorithm is used." << endl << endl;
            VectorXd q(length), r(length), v(length), alpha(solverSize), beta(solverSize), Coord(NDim);
            MatrixXd T(solverSize, solverSize), Q(length, solverSize);
            /* 
                Using Harmonic Oscillator states (Gaussian function times polynomial) 
                as the initial Lanczos vector q. In this way, we want to the subspace includes
                the first several excited states. 

                In saveMem mode, the Hamiltonian and indice is not initialized. 
                It will use H_times_V and oneD2mD instead. 
            */
            // for(int ii = 0; ii < length; ii++)
            // {
            //     q(ii) = 1.0;
            //     for(int dd = 0; dd < NDim; dd++)
            //     {
            //         Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
            //         double tmpEXP = exp( - pow(Coord(dd), 2) / 2.0);
            //         q(ii) = tmpEXP * (1.0 + Coord(dd) + pow(Coord(dd),2) + pow(Coord(dd),3) + pow(Coord(dd), 4) +pow(Coord(dd), 5) );
            //     }
            // }
            q = VectorXd::Random(length);
            q = q/q.norm();
            
            Q = 0.0 * Q;
            T = 0.0 * T;
            #pragma omp parallel for
            for(int kk = 0; kk < length; kk++)  Q(kk, 0) = q(kk);
            r = H_times_V(q);
            alpha(0) = q.transpose() * r;
            r = r -alpha(0) * q;
            beta(0) = r.norm();
            T(0,0) = alpha(0);
            T(0,1) = beta(0);
            T(1,0) = beta(0);
            
            for(int jj = 1; jj < solverSize; jj++)
            {
                v = q;
                q = r/beta[jj - 1];
                #pragma omp parallel for
                for(int kk = 0; kk < length; kk++)  Q(kk, jj) = q(kk);
                r = H_times_V(q) - beta(jj - 1) * v;
                alpha(jj) = q.transpose() * r;
                r = r - alpha(jj) * q;
                r = r - Q.block(0, 0, length, jj)*(Q.block(0, 0, length, jj).transpose()*r);
                beta(jj) = r.norm(); 
                cout << "beta_" << jj + 1 << " = " << beta(jj) << endl;

                T(jj, jj) = alpha(jj);
                if(jj != solverSize - 1)
                {
                    T(jj, jj + 1) = beta(jj);
                    T(jj + 1, jj) = beta(jj);
                }
            }
            
            SelfAdjointEigenSolver<MatrixXd> KernelSolver(T);
            eigenValues = KernelSolver.eigenvalues();
            eigenStates = Q * KernelSolver.eigenvectors();
            energies = eigenValues;
            states = eigenStates;
        }
        else
        {
            if(lanczos)
            {
                cout << "The reorthogonalized Lanczos Algorithm is used." << endl << endl;
                VectorXd q(length), r(length), v(length), alpha(solverSize), beta(solverSize), Coord(NDim);
                MatrixXd T(solverSize, solverSize), Q(length, solverSize);
                /* 
                    Using Harmonic Oscillator states (Gaussian function times polynomial) 
                    as the initial Lanczos vector q. In this way, we want to the subspace includes
                    the first several excited states. 
                */
                // for(int ii = 0; ii < length; ii++)
                // {
                //     q(ii) = 1.0;
                //     for(int dd = 0; dd < NDim; dd++)
                //     {
                //         Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
                //         double tmpEXP = exp( - pow(Coord(dd), 2) / 2.0);
                //         q(ii) = tmpEXP * (1.0 + Coord(dd) + pow(Coord(dd),2) + pow(Coord(dd),3) + pow(Coord(dd), 4) +pow(Coord(dd), 5) );
                //     }
                // }
                q = VectorXd::Random(length);/////
                cout << q(0) << q(10) << q(20) << endl;
                q = q/q.norm();

                Q = 0.0 * Q;
                T = 0.0 * T;
                #pragma omp parallel for
                for(int kk = 0; kk < length; kk++)  Q(kk, 0) = q(kk);
                r = Hamiltonian * q;
                alpha(0) = q.transpose() * r;
                r = r -alpha(0) * q;
                beta(0) = r.norm();
                T(0,0) = alpha(0);
                T(0,1) = beta(0);
                T(1,0) = beta(0);

                for(int jj = 1; jj < solverSize; jj++)
                {
                    v = q;
                    q = r/beta[jj - 1];
                    #pragma omp parallel for
                    for(int kk = 0; kk < length; kk++)  Q(kk, jj) = q(kk);
                    r = Hamiltonian * q - beta(jj - 1) * v;
                    alpha(jj) = q.transpose() * r;
                    r = r - alpha(jj) * q;
                    r = r - Q.block(0, 0, length, jj)*(Q.block(0, 0, length, jj).transpose()*r);
                    beta(jj) = r.norm(); 
                    cout << "beta_" << jj + 1 << " = " << beta(jj) << endl;

                    T(jj, jj) = alpha(jj);
                    if(jj != solverSize - 1)
                    {
                        T(jj, jj + 1) = beta(jj);
                        T(jj + 1, jj) = beta(jj);
                    }
                }

                SelfAdjointEigenSolver<MatrixXd> KernelSolver(T);
                eigenValues = KernelSolver.eigenvalues();
                eigenStates = Q * KernelSolver.eigenvectors();
                energies = eigenValues;
                states = eigenStates;
            }
            else
            {
                if(SYM_EIGEN_D(Hamiltonian, length, energies, states, solverSize) > 0)
                {
                    cout << "SYM_EIGEN_D does not work well." << endl;
                    exit(99);
                }
            }
            
        }
    }
    
    solved = true;
}


/*
    Special evaluation functions used in memory saving purpose.
    oneD2mD(lll) equivalently returns lll th row in indice.
    eltsH(iii, jjj) returns H(iii, jjj) without store an explicit matrix
    H_times_V(V) returns product of Hamiltonian and V.
*/
inline VectorXi DVR::oneD2mD(const int& lll)const
{
    VectorXi indiceMD(NDim);
    int tmpll = lll;
    for(int dd = 0; dd < NDim; dd++)
    {
        int tmp = 1;
        for(int ii = NDim - 1; ii > dd; ii--)
        {
            tmp = tmp * NGrids(ii);
        }
        indiceMD(dd) = tmpll/tmp;
        tmpll = tmpll - indiceMD(dd) * tmp;
    }
    return indiceMD;
}
inline int DVR::mD2oneD(const VectorXi& indicesMD)const
{
    int tmp = 1, returnValue = 0;
	for(int ii = NDim - 1; ii >= 0; ii--)
	{
		returnValue += tmp * indicesMD(ii);
		tmp = tmp * NGrids(ii);
	}

	return returnValue;
}
double DVR::eltsH(const int& iii, const int& jjj)const
{
    double element = 0.0;
    if(iii == jjj) element += Hamiltonian(0, iii);
    #pragma omp parallel for
    for(int dd = 0; dd < NDim; dd++)
    {
        bool Ktmp = true;
        for(int ddtmp = 0; ddtmp < NDim; ddtmp++)
        {
            /* These "if" are the delta symbols in equ.(2.6) */
            if(ddtmp != dd)
            {
                if(indice(iii, ddtmp) != indice(jjj, ddtmp)) Ktmp = false;
            }
        }
        if(Ktmp) element = element + Hamiltonian(dd + 1, abs(indice(iii, dd) - indice(jjj, dd)));
    }
    
    return element;
}
VectorXd DVR::H_times_V(const VectorXd& V)const
{
    if(V.rows() != length)
    {
        std::cout << "The Size of V is inconsistent with H in H_times_V." << std::endl;
        exit(99);
    }
    else
    {
        VectorXd HV(length);
        #pragma omp parallel for
        for(int ii = 0; ii < length; ii++)
        {
            VectorXi indicesMD = indice.block(ii,0,1,NDim).transpose();
            HV(ii) = Hamiltonian(0, ii) * V(ii);
            for(int dd = 0; dd < NDim; dd++)
            {
                double tmp = 0.0;
                VectorXi indicesMDtmp = indicesMD;
                for(int jj = 0; jj < NGrids(dd); jj++)
                {
                    indicesMDtmp(dd) = jj;
                    tmp += Hamiltonian(dd + 1, abs(jj - indicesMD(dd))) * V(mD2oneD(indicesMDtmp));
                }
                HV(ii) += tmp;
            }
        }
        return HV;
    }
}


/*
    Public functions to get corresponding private matrices
*/
MatrixXd DVR::getHamiltonian()
{
    return Hamiltonian;
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