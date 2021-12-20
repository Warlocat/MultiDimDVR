#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include<Eigen/Dense>
#include<iostream>
#include<float.h>
#include"DVR.h"
#include"fitting.h"
#include<string>
#include<cmath>
#include<iomanip>
#include<fstream>
using namespace std;
using namespace Eigen;

double m1 = 88.9058, m2 = 15.9990;
const double ATOMIC_MASS_UNIT = 1.660539040/9.1093826*10000, au2cm_1 = 219474.63, ang2bohr = 1.8897161646320724;

double factorial(const int& n), eminTMP = 100, bondLength;
Matrix<MatrixXi, -1, 1> PESparametersIJ;
Matrix<VectorXd, -1, 1> PESparameters;
void readPESparameters(const int& ND, const string& filename, const int& index = 0);
void readInput(int& NDim, VectorXi& NGrids, VectorXd& CoordStart, VectorXd& CoordEnd, int& lanczos_iter);
void writeOutput(const MatrixXd& energies, const MatrixXd& states, const int& Nout = 50);
double scanBondLength(MatrixXd (* PotentialPointer_)(const VectorXd& Coord, const int& ND));

MatrixXd potentialFunction0(const VectorXd& Coord, const int& ND);
MatrixXd potentialFunction1(const VectorXd& Coord, const int& ND);
MatrixXd potentialFunctionMatrix(const VectorXd& Coord, const int& ND);

MatrixXd evaluateFCF(const MatrixXd& states1, const MatrixXd& states2, const int& num1, const int& num2);
MatrixXd evaluateMatrixElement(const MatrixXd& states1, const MatrixXd& states2, const int& num1, const int& num2, MatrixXd (* PotentialPointer_)(const VectorXd& Coord, const int& ND), const DVR& dvrInstant);
Vector2d evaluateOmega_eChi_e(const VectorXd& values, const int& num);


int main()
{
    int NDim, fullDim = 1, nstates = 2, lanczos_iter;
    bool saveMem = false, readPES = false, useLanczos = false;
    VectorXi NGrids;
    VectorXd CoordStart, CoordEnd, mass, omega, fullomega(fullDim);
    Matrix<VectorXd, -1, 1> energies;
    Matrix<MatrixXd, -1, 1> states;
    
    readInput(NDim, NGrids, CoordStart, CoordEnd, lanczos_iter);
    
    PESparameters.resize(nstates);
    PESparametersIJ.resize(nstates);
    energies.resize(nstates);
    states.resize(nstates);
    readPESparameters(NDim, "fitV0", 0);
    readPESparameters(NDim, "fitV1", 1);
    mass.resize(NDim);
    for(int ii = 0; ii < NDim; ii++)  mass(ii) = m1*m2 * ATOMIC_MASS_UNIT / (m1+m2);

    DVR dvr_state0(NDim, NGrids, CoordStart, CoordEnd, mass, potentialFunction0, 1, saveMem, readPES);
    DVR dvr_state1(NDim, NGrids, CoordStart, CoordEnd, mass, potentialFunction1, 1, saveMem, readPES);
    
    dvr_state0.kernel(energies(0), states(0));
    dvr_state1.kernel(energies(1), states(1));

    writeOutput(energies(0), states(0), 8);
    writeOutput(energies(1), states(1), 8);

    // MatrixXd fcf = evaluateFCF(states(0), states(1), 10, 1);
    MatrixXd fcf = evaluateMatrixElement(states(0), states(1), 10, 1, potentialFunctionMatrix, dvr_state0);

    for(int ii = 0; ii < fcf.cols(); ii++)
    {
        for(int jj = 0; jj < fcf.rows(); jj++)
            cout << fixed << setprecision(5) << fcf(jj,ii) << endl;
        cout << endl;
    }
    

    scanBondLength(potentialFunction0);
    cout << "Freq and anh 0: " << endl << evaluateOmega_eChi_e(energies(0),8) << endl;
    scanBondLength(potentialFunction1);
    cout << "Freq and anh 1: " << endl << evaluateOmega_eChi_e(energies(1),8) << endl;

    cout << "Program finished normally." << endl;

    return 0;
}



MatrixXd potentialFunction0(const VectorXd& Coord, const int& ND)
{
    MatrixXd V(1,1);
    /* harmonic */
    // double req = 1.7909, omega = 853.5;
    // double tmp_d = Coord(0) - req * ang2bohr;
    // V(0,0) = 0.5 *  m1*m2 * ATOMIC_MASS_UNIT / (m1+m2) * tmp_d*tmp_d * omega*omega / au2cm_1 / au2cm_1;

    /* Morse */
    double req = 1.7909, omega = 903.5 / au2cm_1, omega_chi = 2.7 / au2cm_1;
    double D = omega*omega/omega_chi/4.0, a = sqrt(m1*m2 * ATOMIC_MASS_UNIT / (m1+m2) * omega*omega / 2.0 / D);
    double tmp_d = Coord(0) - req * ang2bohr;
    V(0,0) = D * pow(1.0 - exp(-a * tmp_d),2);
    
    /* ab initio */
    // double coord = Coord(0)  - (0.0 - 0.0) * ang2bohr;
    // double V = 0.0;
    // double req = 1.8;
    // double tmp_d = coord - req * ang2bohr;
    // for(int ii = 0; ii < PESparameters(0).rows(); ii++) 
    // {
    //     double tmp = 1.0;
    //     for(int jj = 0; jj < ND; jj++)
    //     {
    //         tmp = tmp * pow(tmp_d, PESparametersIJ(0)(ii, jj));
    //     }
    //     V = V + tmp * PESparameters(0)(ii);
    // }
    // V(0,0) = V / au2cm_1;

    return V;
}
MatrixXd potentialFunction1(const VectorXd& Coord, const int& ND)
{
    MatrixXd V(1,1);
    /* harmonic */
    // double req = 1.8198, omega = 785.7;
    // double tmp_d = Coord(0) - req * ang2bohr;
    // V(0,0) = 0.5 *  m1*m2 * ATOMIC_MASS_UNIT / (m1+m2) * tmp_d*tmp_d * omega*omega / au2cm_1 / au2cm_1;

    /* Morse */
    double req = 1.8252, omega = 774.6 / au2cm_1, omega_chi = 2.9 / au2cm_1;
    double D = omega*omega/omega_chi/4.0, a = sqrt(m1*m2 * ATOMIC_MASS_UNIT / (m1+m2) * omega*omega / 2.0 / D);
    double tmp_d = Coord(0) - req * ang2bohr;
    V(0,0) = D * pow(1.0 - exp(-a * tmp_d),2);

    /* ab initio */
    // double coord = Coord(0)  - (0.0 - 0.0) * ang2bohr;
    // double V = 0.0;
    // double req = 1.8;
    // double tmp_d = coord- req * ang2bohr;
    // for(int ii = 0; ii < PESparameters(1).rows(); ii++) 
    // {
    //     double tmp = 1.0;
    //     for(int jj = 0; jj < ND; jj++)
    //     {
    //         tmp = tmp * pow(tmp_d, PESparametersIJ(1)(ii, jj));
    //     }
    //     V = V + tmp * PESparameters(1)(ii);
    // }
    // V(0,0) = V / au2cm_1;

    return V;
}
MatrixXd potentialFunctionMatrix(const VectorXd& Coord, const int& ND)
{
    MatrixXd V(1,1);
    V(0,0) = 1.0;

    return V;
}


double factorial(const int &n)
{
    double fac = 1.0;
    if(n == 0 || n == 1) return 1;
    for(int ii = 2; ii <=n; ii++)
    {
        fac = fac * ii;
    }
    return fac;
}

void readPESparameters(const int& ND, const string& filename, const int& index)
{
    int N;
    double tmp;
    ifstream ifs;
    ifs.open(filename);
        ifs >> N;
        PESparametersIJ(index).resize(N,ND);
        PESparameters(index).resize(N);
        for(int ii = 0; ii < N; ii++) 
        {
            for(int jj = 0; jj < ND; jj++) 
            {
                ifs >> PESparametersIJ(index)(ii, jj);
            }
            ifs >> PESparameters(index)(ii);
            for(int jj = 0; jj < PESparametersIJ(index).cols(); jj++)
            {
                PESparameters(index)(ii) = PESparameters(index)(ii) / factorial(PESparametersIJ(index)(ii,jj));
            }
        }
    ifs.close();
}


void readInput(int& NDim, VectorXi& NGrids, VectorXd& CoordStart, VectorXd& CoordEnd, int& lanczos_iter)
{
    ifstream ifs;
    string flags;
    ifs.open("input");
        ifs >> NDim >> flags;
        NGrids.resize(NDim);
        CoordStart.resize(NDim);
        CoordEnd.resize(NDim);
        for(int ii = 0; ii < NDim; ii++)
        {
            ifs >> CoordStart(ii) >> CoordEnd(ii) >> NGrids(ii);
        }
        ifs >> lanczos_iter >> flags;
    ifs.close();

    return;
}

void writeOutput(const MatrixXd& energies, const MatrixXd& states, const int& Nout)
{
    double tmp = 0.0;
    for(int ii = 0; ii < Nout; ii++)    
    {
        tmp = 0.0;
        // cout << fixed << setprecision(2) << (energies(ii) - energies(0))*au2cm_1 << endl;
        cout << setprecision(16) << energies(ii) << endl;
    }
    return;
}

MatrixXd evaluateFCF(const MatrixXd& states1, const MatrixXd& states2, const int& num1, const int& num2)
{
    if(states1.rows() != states2.rows())
    {
        cout << "ERROR: states1 and state2 have different dimension\n";
        exit(99);
    }
    MatrixXd FCF(num1,num2);
    FCF = MatrixXd::Zero(num1,num2);
    for(int ii = 0; ii < num1; ii++)
    for(int jj = 0; jj < num2; jj++)
    {
        for(int kk = 0; kk < states1.rows(); kk++)
            FCF(ii,jj) += states1(kk,ii) * states2(kk,jj);
        FCF(ii,jj) = pow(FCF(ii,jj),2);
    }

    return FCF;
}

MatrixXd evaluateMatrixElement(const MatrixXd& states1, const MatrixXd& states2, const int& num1, const int& num2, MatrixXd (* PotentialPointer_)(const VectorXd& Coord, const int& ND), const DVR& dvrInstant)
{
    if(states1.rows() != states2.rows())
    {
        cout << "ERROR: states1 and state2 have different dimension\n";
        exit(99);
    }
    MatrixXd FCF(num1,num2);
    VectorXd coord(dvrInstant.CoordStart.rows());
    FCF = MatrixXd::Zero(num1,num2);
    for(int ii = 0; ii < num1; ii++)
    for(int jj = 0; jj < num2; jj++)
    {
        for(int kk = 0; kk < states1.rows(); kk++)
        {
            for(int dd = 0; dd < coord.rows(); dd++)
            {
                coord(dd) = dvrInstant.CoordStart(dd) + dvrInstant.dx(dd)*dvrInstant.oneD2mD(kk)(dd);
            }
            FCF(ii,jj) += states1(kk,ii) * states2(kk,jj) * PotentialPointer_(coord,dvrInstant.CoordStart.rows())(0,0);
        }
        FCF(ii,jj) = pow(FCF(ii,jj),2);
    }

    return FCF;
}


double scanBondLength(MatrixXd (* PotentialPointer_)(const VectorXd& Coord, const int& ND))
{
    double x0 = 1.7, dx = 0.0001, Emin = 1e20, bl;
    for(int ii = 0; ii < 2000; ii++)
    {
        double xtmp = x0 + ii * dx;
        VectorXd coordtmp(1);
        coordtmp(0) = xtmp * ang2bohr;
        double etmp = PotentialPointer_(coordtmp,1)(0,0);
        if(etmp < Emin)
        {
            bl = xtmp;
            Emin = etmp;
        }
    }
    cout << "Bond length: " << bl << ", and term energy: " << setprecision(16) << Emin << endl; 
    return bl;
}

Vector2d evaluateOmega_eChi_e(const VectorXd& values, const int& num)
{
    Vector2d Omega_eChi_e;
    VectorXd X(num-1), Y(num-1);
    MatrixXd A(3,3), B(3,1), C;
    for(int ii = 0; ii < num-1; ii++)
    {
        X(ii) = ii+1;
        Y(ii) = (values(ii+1) - values(0)) * 219474.63;
    }
    for(int ii = 0; ii < 3; ii++)
    {
        B(ii) = 0;
        for(int kk = 0; kk < num-1; kk++)
            B(ii) += pow(X(kk),ii) * Y(kk);
        for(int jj = 0; jj < 3; jj++)
        {
            A(ii,jj) = 0.0;
            for(int kk = 0; kk < num-1; kk++)
                A(ii,jj) += pow(X(kk),ii) * pow(X(kk),jj);
        }
    }
    C = A.inverse() * B;
    Omega_eChi_e(1) = -C(2);
    Omega_eChi_e(0) = C(1) + Omega_eChi_e(1);

    return Omega_eChi_e;
}
