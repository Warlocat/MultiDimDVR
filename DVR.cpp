#include"DVR.h"
#include<iostream>
#include<fstream>
#include<Eigen/Dense>
#include<omp.h>
using namespace Eigen;
using namespace std;

/* 
    Construction function with initialization 
*/
DVR::DVR(const int& NDim_, const VectorXi& NGrids_, const VectorXd& CoordStart_, const VectorXd& CoordEnd_, const VectorXd& mass_, MatrixXd (* PotentialPointer_)(const VectorXd& Coord, const int& ND), const int& NStates_, const bool &saveMem_, const bool& readPESfromFile_)
: NDim(NDim_), NGrids(NGrids_), CoordStart(CoordStart_), saveMem(saveMem_), 
CoordEnd(CoordEnd_), mass(mass_), PotentialPointer(PotentialPointer_), NStates(NStates_), readPESfromFile(readPESfromFile_)
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
    if(iii == jjj) kinetics = 1.0 / 2.0 / massX / pow(deltaX, 2) * (M_PI * M_PI / 3.0);
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
            When saveMem mode is on. kineticT and potentialV will not be constructed explicitly. 
            Instead, they will be resized to a small matrix.
            The potentialV stores the potential energies.
            The kineticT stores the kinetic energies for different dimension using the delta iii - jjj.
        */
        int size_nstates = NStates * (NStates + 1) / 2; 
        kineticT.resize(NDim, length * NStates);
        potentialV.resize(size_nstates, length * NStates);
        if(readPESfromFile)
        {
            if(NStates > 1)
            {
                cout << "When more than one states is included, reading PES from file is NOT supported." << endl;
                exit(99);
            }
            ifstream ifs;
            int tmp;
            ifs.open("PESinput");
            ifs >> tmp;
            if(tmp != length)
            {
                cout << "data in PESinput does not match the Hamiltonian size";
                ifs.close();
                exit(99);
            }
            else
            {
                for(int ii = 0; ii < length; ii++)
                {
                    ifs >> potentialV(0, ii);
                }
            }
            ifs.close();
        }
        for(int ii = 0; ii < length; ii++)
        {
            for(int dd = 0; dd < NDim; dd++)
            {
                Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
                kineticT(dd, ii) = oneDimK(dx(dd), mass(dd), 0, ii);
            }
            if(!readPESfromFile) 
            {
                for(int mm = 0; mm < NStates; mm++)
                for(int nn = 0; nn <= mm; nn++)
                {
                    potentialV(mm * (mm+1) / 2 + nn, ii) = PotentialPointer(Coord, NDim)(mm,nn);
                }
            }
        }
    }
    else
    {
        kineticT.resize(length * NStates, length * NStates);
        potentialV.resize(length * NStates, length * NStates);
        kineticT = MatrixXd::Zero(length * NStates, length * NStates);
        potentialV = MatrixXd::Zero(length * NStates, length * NStates);

        /* Calculate potentialE and kineticE matrices, add them together */ 
        if(readPESfromFile)
        {
            if(NStates > 1)
            {
                cout << "When more than one states is included, reading PES from file is NOT supported." << endl;
                exit(99);
            }
            ifstream ifs;
            int tmp;
            ifs.open("PESinput");
            ifs >> tmp;
            if(tmp != length)
            {
                cout << "data in PESinput does not match the Hamiltonian size";
                ifs.close();
                exit(99);
            }
            else
            {
                for(int ii = 0; ii < length; ii++)
                {
                    ifs >> potentialV(ii, ii);
                }
            }
            ifs.close();
        }
        for(int ii = 0; ii < length; ii++)
        {
            if(!readPESfromFile)
            {
                for(int dd = 0; dd < NDim; dd++)
                {
                    Coord(dd) = CoordStart(dd) + indice(ii, dd) * dx(dd);
                }
                MatrixXd pesTMP = PotentialPointer(Coord, NDim);

                /* This is correct only for diabatic Hamiltonian */
                for(int ss1 = 0; ss1 < NStates; ss1++)
                for(int ss2 = 0; ss2 < NStates; ss2++)
                {
                    potentialV(ss1*length + ii, ss2*length + ii) = pesTMP(ss1, ss2);
                }
                
                // std::cout << Coord(0) << "\t" << potentialE(ii, ii) << std::endl;
            }
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
                for(int ss = 0; ss < NStates; ss++)
                {
                    kineticT(ss*length + ii, ss*length + jj) = tmp;
                }
                // Hamiltonian(ii, jj) += tmp;
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
    SelfAdjointEigenSolver<MatrixXd> KernelSolver(kineticT + potentialV);
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

        if(!lanczos) cout << endl << "saveMem mode is on. Automatically switch to Lanczos Algorithm." << solverSize << endl;
        cout << "The reorthogonalized Lanczos Algorithm is used." << endl << endl;
        VectorXd q(length * NStates), r(length * NStates), v(length * NStates), alpha(solverSize), beta(solverSize), Coord(NDim);
        MatrixXd T(solverSize, solverSize), Q(length * NStates, solverSize);
        /* 
            Using Harmonic Oscillator states (Gaussian function times polynomial) 
            as the initial Lanczos vector q. In this way, we want to the subspace includes
            the first several excited states. 
        */
        q = VectorXd::Random(length * NStates);
        q = q/q.norm();
        
        Q = 0.0 * Q;
        T = 0.0 * T;
        #pragma omp parallel for
        for(int kk = 0; kk < length * NStates; kk++)  Q(kk, 0) = q(kk);
        H_times_V(q, r);
        alpha(0) = q.adjoint() * r;
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
            for(int kk = 0; kk < length * NStates; kk++)  Q(kk, jj) = q(kk);
            H_times_V(q, r);
            r = r - beta(jj - 1) * v;
            alpha(jj) = q.adjoint() * r;
            r = r - alpha(jj) * q;
            r = r - Q.block(0, 0, length * NStates, jj)*(Q.block(0, 0, length * NStates, jj).adjoint()*r);
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
    
    solved = true;
}


/*
    Special evaluation functions used in memory saving purpose.
    oneD2mD(lll) equivalently returns lll th row in indice.
    H_times_V(V) returns product of Hamiltonian and V.
*/
VectorXi DVR::oneD2mD(const int& lll)const
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
int DVR::mD2oneD(const VectorXi& indicesMD)const
{
    int tmp = 1, returnValue = 0;
	for(int ii = NDim - 1; ii >= 0; ii--)
	{
		returnValue += tmp * indicesMD(ii);
		tmp = tmp * NGrids(ii);
	}

	return returnValue;
}

void DVR::H_times_V(const VectorXd& V, VectorXd& result)const
{
    if(V.rows() != length * NStates)
    {
        std::cout << "The Size of V is inconsistent with H in H_times_V." << std::endl;
        exit(99);
    }
    else
    {
        if(!saveMem)
        {
            result = (kineticT + potentialV) * V;
        }
        else
        {
            result.resize(length * NStates);
            result = VectorXd::Zero(length * NStates);
            for(int ss = 0; ss < NStates; ss++)
            {
                #pragma omp parallel for
                for(int ii = 0; ii < length; ii++)
                {
                    VectorXi indicesMD = indice.block(ii,0,1,NDim).transpose();
                    for(int rr = 0; rr < NStates; rr++)
                    {
                        if(ss >= rr) result(ii + ss*length) += potentialV(ss * (ss+1) / 2 + rr, ii) * V(ii + rr*length);
                        else result(ii + ss*length) += potentialV(rr * (rr+1) / 2 + ss, ii) * V(ii + rr*length);
                    }

                    for(int dd = 0; dd < NDim; dd++)
                    {
                        double tmp = 0.0;
                        VectorXi indicesMDtmp = indicesMD;
                        for(int jj = 0; jj < NGrids(dd); jj++)
                        {
                            indicesMDtmp(dd) = jj;
                            tmp += kineticT(dd, abs(jj - indicesMD(dd))) * V(mD2oneD(indicesMDtmp) + ss*length);
                        }
                        result(ii + ss*length) += tmp;
                    }
                }
            }   
        }
    }
}


/*
    Public functions to get corresponding private matrices
*/
MatrixXd DVR::getHamiltonian()
{
    return kineticT + potentialV;
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
