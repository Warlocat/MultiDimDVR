#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include<Eigen/Dense>
#include<iostream>
#include<float.h>
#include"DVR_C.h"
#include"DVR.h"
#include<string>
#include<complex>
#include<cmath>
#include<iomanip>
#include<fstream>
using namespace std;
using namespace Eigen;

const double ATOMIC_MASS_UNIT = 1.660539040/9.1093826*10000, au2cm_1 = 219474.63, ang2bohr = 1.8897161646320724;
const dcomplex III(0.0,1.0);
MatrixXd refCoord(3,3);
Matrix<Matrix3d, 4, 1> Qall;
Matrix<Matrix3d, -1, 1> Q;

double factorial(const int& n);
Matrix<MatrixXi, -1, 1> PESparametersIJ;
Matrix<VectorXd, -1, 1> PESparameters;
void read_QUADRATURE(const string& filename, Matrix<Matrix3d, 4, 1>& Qall, MatrixXd& refCoord, VectorXd& omega, const int& ND);
void readPESparameters(const int& ND, const string& filename, const int& index = 0);
void readInput(int& NDim, VectorXi& NGrids, VectorXd& CoordStart, VectorXd& CoordEnd, int& lanczos_iter);
void writeOutput(const VectorXcd& energies, const MatrixXcd& states, const int& Nout = 20);

VectorXd normal2internal_A(const VectorXd& Coord, const int& ND);
VectorXd normal2internal_B(const VectorXd& Coord, const int& ND);
MatrixXcd potentialFunction(const VectorXd& Coord, const int& ND);
double potentialFunction_signle_wn(const VectorXd& Coord, const int& ND, const int& index);

int main()
{
    int NDim, nstates = 3, fullDim = 4, lanczos_iter;
    bool saveMem = true, readPES = false, useLanczos = true;
    VectorXi NGrids;
    VectorXd CoordStart, CoordEnd, mass, omega, fullomega(fullDim);
    VectorXcd energies;
    MatrixXcd states;
    
    readInput(NDim, NGrids, CoordStart, CoordEnd, lanczos_iter);
    read_QUADRATURE("QUADRATURE_pi1", Qall, refCoord, fullomega, fullDim);
    PESparameters.resize(nstates);
    PESparametersIJ.resize(nstates);
    readPESparameters(NDim, "pi1_fit_6", 0);  
    readPESparameters(NDim, "pi2_fit_6", 1);  
    readPESparameters(NDim, "sigmaB_6", 2);  
    MatrixXcd (* PotentialPointer)(const VectorXd& Coord, const int& ND) = potentialFunction;
    
    Q.resize(NDim);
    mass.resize(NDim);
    omega.resize(NDim);
    fullomega = fullomega / au2cm_1;
    Q << Qall(0), Qall(1), Qall(2), Qall(3);
    omega << fullomega(0), fullomega(1), fullomega(2), fullomega(3);
    //Q << Qall(0)i, Qall(1);
   // omega << fullomega(0), fullomega(1);
    for(int ii = 0; ii < NDim; ii++)  mass(ii) = 1.0 / omega(ii);

    DVR_C dvr_state1(NDim, NGrids, CoordStart, CoordEnd, mass, potentialFunction, nstates, saveMem, readPES);
    dvr_state1.kernel(energies, states, lanczos_iter, useLanczos);

    writeOutput(energies, states);
    cout << "Program finished normally." << endl;

    return 0;
}



MatrixXcd potentialFunction(const VectorXd& Coord, const int& ND)
{
    double xx = Coord(0), yy = Coord(1);
    double Epp = potentialFunction_signle_wn(Coord, ND, 0);
    double Emm = potentialFunction_signle_wn(Coord, ND, 1);
    double Ebb = potentialFunction_signle_wn(Coord, ND, 2);
    double lambdaX = 80.0, lambdaY = 80.0, coupling, correction;
    coupling = lambdaX * xx + lambdaY * yy;
    correction = sqrt((Ebb-Epp)*(Ebb-Epp) - 4*coupling*coupling);
    
    Matrix3d V1;
    Matrix3cd V2;
    V1 <<   0.5*(Epp+Ebb-correction), 0.0, coupling, 
            0.0, Emm, 0.0, 
            coupling, 0.0, 0.5*(Epp+Ebb+correction);
    double c1 = xx/sqrt(xx*xx+yy*yy);
    double c2 = yy/sqrt(xx*xx+yy*yy);
    if(abs(xx) <1e-5 && abs(yy) < 1e-5)
    {
        c1 = 1.0; c2 = 0.0;
    }
    V2(0,0) = c1*c1*V1(0,0) + c2*c2*V1(1,1);
    V2(1,1) = c2*c2*V1(0,0) + c1*c1*V1(1,1);
    V2(2,2) = V1(2,2);
    V2(1,0) = c1*c2*(V1(0,0) - V1(1,1)) + 130.3*III;
    V2(2,0) = c1 * coupling;
    V2(2,1) = -c2 * coupling;
    V2(0,1) = conj(V2(1,0));
    V2(0,2) = conj(V2(2,0));
    V2(1,2) = conj(V2(2,1));
    return V2;
}


double potentialFunction_signle_wn(const VectorXd& Coord, const int& ND, const int& index)
{
    double V = 0.0;
    VectorXd internal;
    if(index == 2) internal = normal2internal_B(Coord, ND);
    else internal = normal2internal_A(Coord, ND);

    for(int ii = 0; ii < PESparameters(index).rows(); ii++) 
    {
        double tmp = 1.0;
        for(int jj = 0; jj < 3; jj++)
        {
            tmp = tmp * pow(internal(jj), PESparametersIJ(index)(ii, jj));
        }
        V = V + tmp * PESparameters(index)(ii);
    }

    return V;
}




void read_QUADRATURE(const string& filename, Matrix<Matrix3d, 4, 1>& Qall, MatrixXd& refCoord, VectorXd& omega, const int& ND)
{
    ifstream ifs;
    string flags;
    ifs.open(filename);
        for(int ii = 0; ii < ND; ii++)
        {
            ifs >> flags >> flags >> flags >> flags >> omega(ii);
            ifs >> flags >> flags >> flags >> flags >> flags >> flags >> flags;
            for(int jj = 0; jj < 3; jj++)
            for(int kk = 0; kk < 3; kk++)
            {
                ifs >> Qall(ii)(jj, kk);
            }  
        }
        ifs >> flags >> flags >> flags >> flags >> flags;
        for(int jj = 0; jj < 3; jj++)
        for(int kk = 0; kk < 3; kk++)
        {
            ifs >> refCoord(jj, kk);
        }
	
    ifs.close();
}


VectorXd normal2internal_A(const VectorXd& Coord, const int& ND)
{
    double tmp;
    Vector3d tmp1, tmp2, internal, r1e, r2e;
    Matrix3d cart = refCoord;
    for(int ii = 0; ii < ND; ii++)
    {
        cart += Coord(ii) * Q(ii);
    }

    for(int ii = 0; ii < 3; ii++)
    {
        tmp1(ii) = cart(0, ii) - cart(1, ii);
        tmp2(ii) = cart(2, ii) - cart(1, ii);
        r1e(ii) = refCoord(1, ii) - refCoord(0, ii);
        r2e(ii) = refCoord(2, ii) - refCoord(1, ii);
    }
    // internal(0) = tmp1.norm() - r1e.norm();
    // internal(1) = tmp2.norm() - r2e.norm();
    internal(0) = tmp1.norm() - 3.93646667;
    //internal(1) = tmp2.norm() - 1.84737507;
    internal(1) = tmp2.norm() - 1.79737507;
    tmp = tmp1.transpose() * tmp2;
    tmp = tmp / tmp1.norm() / tmp2.norm();
    if(tmp > 1.0) tmp = 1.0;
    else if (tmp < -1.0) tmp = -1.0;

    internal(2) = acos(tmp) / M_PI * 180.0 - 180.0;
    // cout << internal.transpose() << endl;
    return internal;
}
VectorXd normal2internal_B(const VectorXd& Coord, const int& ND)
{
    double tmp;
    Vector3d tmp1, tmp2, internal, r1e, r2e;
    Matrix3d cart = refCoord;
    for(int ii = 0; ii < ND; ii++)
    {
        cart += Coord(ii) * Q(ii);
    }

    for(int ii = 0; ii < 3; ii++)
    {
        tmp1(ii) = cart(0, ii) - cart(1, ii);
        tmp2(ii) = cart(2, ii) - cart(1, ii);
        r1e(ii) = refCoord(1, ii) - refCoord(0, ii);
        r2e(ii) = refCoord(2, ii) - refCoord(1, ii);
    }
    internal(0) = tmp1.norm() - 3.94722723;
    internal(1) = tmp2.norm() - 1.79641517;
    tmp = tmp1.transpose() * tmp2;
    tmp = tmp / tmp1.norm() / tmp2.norm();
    if(tmp > 1.0) tmp = 1.0;
    else if (tmp < -1.0) tmp = -1.0;

    internal(2) = acos(tmp) / M_PI * 180.0 - 180.0;
    // cout << internal.transpose() << endl;
    return internal;
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
        PESparametersIJ(index).resize(N,3);
        PESparameters(index).resize(N);
        for(int ii = 0; ii < N; ii++) 
        {
            for(int jj = 0; jj < 3; jj++) 
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

void writeOutput(const VectorXcd& energies, const MatrixXcd& states, const int& Nout)
{
    ofstream ofs1, ofs2;
    ofs1.open("ENERGY");
    ofs2.open("STATES", ios::binary);
        ofs1<< (real(energies(0)) + 3253.515147316093) * au2cm_1 << endl; 
        for(int ii = 0; ii < Nout; ii++)    ofs1 << (real(energies(ii) - energies(0))) * au2cm_1 << endl;
        for(int ii = 0; ii < Nout; ii++)    ofs1 << states(ii,0)<< endl;
        for(int ii = 0; ii < states.rows(); ii ++)
        {
            for(int jj = 0; jj < Nout; jj++)
            {
                ofs2.write((char*) &states(ii, jj), sizeof(complex<double>));
            }
        }
    ofs2.close();
    ofs1.close();
}
