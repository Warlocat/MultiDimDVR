#include<iostream>
#include<complex>
#include<fstream>
#include<iomanip>
#include<string>
#include<Eigen/Dense>

using namespace Eigen;
using namespace std;


void readInput(int& NDim, int& NState1, int& NState2, int& NOut1, int& NOut2, int& printLevel, VectorXi& NGrids, Matrix<Vector3d,-1,-1>& transitionDipole, string& fileName1, string& fileName2);
void readEnergy(string filename, double* ene, int nout);
void readWFN(string filename, complex<double>** wfn, int size, int nout);
VectorXi oneD2mD(const int& lll, const int& NDim, const VectorXi& NGrids);
int mD2oneD(const VectorXi& indicesMD, const int& NDim, const VectorXi& NGrids);

int main()
{
    int NDim, NState1, NState2, NOut1, NOut2, printLevel;
    VectorXi NGrids;
    Matrix<Vector3d,-1,-1> transitionDipole;
    string file1, file2; 
    readInput(NDim,NState1,NState2,NOut1,NOut2,printLevel,NGrids,transitionDipole,file1,file2);
    
    int size = 1;    
    for(int ii = 0; ii < NDim; ii++) 
    {
        size = size * NGrids(ii);
    }
    double *ene1, *ene2;
    ene1 = new double[NOut1];
    ene2 = new double[NOut2];
    complex<double> **wfn1, **wfn2;
    wfn1 = new complex<double>* [NState1*size];
    for(int ii = 0; ii < NState1*size; ii++)
        wfn1[ii] = new complex<double>[NOut1];
    wfn2 = new complex<double>* [2*NState2*size];
    for(int ii = 0; ii < 2*NState2*size; ii++)
        wfn2[ii] = new complex<double>[NOut2];
    double br[NOut1][NOut2];
    
    readEnergy("ENERGY_" + file1, ene1, NOut1);
    readEnergy("ENERGY_" + file2, ene2, NOut2);
    readWFN("STATES_" + file1, wfn1, size*NState1, NOut1);
    readWFN("STATES_" + file2, wfn2, size*NState2*2, NOut2);
    
    for(int ii = 0; ii < NOut1; ii++)
    for(int jj = 0; jj < NOut2; jj++)
    {
        double normapx = 0,normapy = 0,normab = 0;
        complex<double> tmp1 =0.0, tmp2=0.0, tmp3=0.0;
        Vector3cd tmp_vc, tmp_vc2;
        Vector3d XX(1.0,0.0,0.0), YY(0.0,1.0,0.0);
        tmp_vc = VectorXcd::Zero(3);
        tmp_vc2 = VectorXcd::Zero(3);
        for(int kk = 0; kk < size; kk++)
        {
            normapx += norm(wfn2[kk][jj]);
            normapy += norm(wfn2[kk+size][jj]);
            normab += norm(wfn2[kk+2*size][jj]);
            tmp_vc += conj(wfn1[kk][ii]) * wfn2[kk][jj] * transitionDipole(0,0);
            tmp_vc += conj(wfn1[kk][ii]) * wfn2[kk+size][jj] * transitionDipole(0,1);
            tmp_vc += conj(wfn1[kk][ii]) * wfn2[kk+2*size][jj] * transitionDipole(0,2);
            tmp_vc2 += conj(wfn1[kk][ii]) * wfn2[kk+3*size][jj] * transitionDipole(0,0);
            tmp_vc2 += conj(wfn1[kk][ii]) * wfn2[kk+4*size][jj] * transitionDipole(0,1);
            tmp_vc2 += conj(wfn1[kk][ii]) * wfn2[kk+5*size][jj] * transitionDipole(0,2);
        }
		if(printLevel>=1)
            cout << ii << "\t" << jj << endl << norm(tmp_vc(0)) << "\t" << norm(tmp_vc(1)) << "\t" << norm(tmp_vc(2)) << "\t" << norm(tmp_vc2(0)) << "\t" << norm(tmp_vc2(1)) << "\t" << norm(tmp_vc2(2)) << endl;   
        br[ii][jj] = 0.5*(pow(tmp_vc.norm(),2) + pow(tmp_vc2.norm(),2)) * pow(ene2[jj] - ene1[ii],3);
    }

    for(int ii = 0; ii < NOut2; ii++)
    {
        double tmp = 0;
        for(int jj = 0; jj < NOut1; jj++)
        {
            tmp += br[jj][ii];
        }
        for(int jj = 0; jj < NOut1; jj++)
        {
            br[jj][ii] /= tmp;
        }
        cout << ii << endl;
        for(int jj = 0; jj < NOut1; jj++)
        {
            cout << br[jj][ii] << endl;
        }
        cout<<endl;
    }

    return 0;

}



void readInput(int& NDim, int& NState1, int& NState2, int& NOut1, int& NOut2, int& printLevel, VectorXi& NGrids, Matrix<Vector3d,-1,-1>& transitionDipole, string& fileName1, string& fileName2)
{
    ifstream ifs;
    string flags;
    ifs.open("input");
        ifs >> NDim >> flags;
        ifs >> NState1 >> flags;
        ifs >> NState2 >> flags;
        ifs >> NOut1 >> flags;
        ifs >> NOut2 >> flags;
		ifs >> printLevel >> flags;
        NGrids.resize(NDim);
        transitionDipole.resize(NState1,NState2);
        for(int ii = 0; ii < NDim; ii++)
        {
            ifs >> NGrids(ii);
        }
        for(int ii = 0; ii < NState1; ii++)
        for(int jj = 0; jj < NState2; jj++)
        {
            ifs >> transitionDipole(ii,jj)(0) >> transitionDipole(ii,jj)(1) >> transitionDipole(ii,jj)(2);
        }
        ifs >> fileName1;
        ifs >> fileName2;
    ifs.close();

    cout << "Reading input:" << endl;
    cout << "Dimension: " << NDim << endl;
    cout << "# States: " << NState1 << "," << NState2 << endl;
    cout << "# Roots: " << NOut1 << "," << NOut2 << endl;
    cout << "Print Level: " << printLevel << endl;
    cout << "# Grid points: " << NGrids.transpose() << endl;
    cout << "Transition Dipole Matrix: " << endl;
    for(int ii = 0; ii < NState1; ii++)
    for(int jj = 0; jj < NState2; jj++)
    {
        cout << ii << "," << jj << "," << transitionDipole(ii,jj).transpose() << endl;
    }

    return;
}

void readEnergy(string filename, double* ene, int nout)
{
    double ene_zero, tmp_d;
    ifstream ifs;
    ifs.open(filename);
    ifs >> ene_zero;
    for(int ii = 0; ii < nout; ii++)
    {
        ifs >> tmp_d;
        ene[ii] = tmp_d + ene_zero;
    }
    ifs.close();
    return;
}
void readWFN(string filename, complex<double>** wfn, int size, int nout)
{
    ifstream ifs;
    ifs.open(filename, ios::binary);
    for(int ii = 0; ii < size; ii ++)
    {
        for(int jj = 0; jj < nout; jj++)
        {
            ifs.read((char*) &wfn[ii][jj], sizeof(complex<double>));
        }
    }
    ifs.close();
    return;
}



/*
    Special evaluation functions used in memory saving purpose.
*/
VectorXi oneD2mD(const int& lll, const int& NDim, const VectorXi& NGrids)
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
int mD2oneD(const VectorXi& indicesMD, const int& NDim, const VectorXi& NGrids)
{
    int tmp = 1, returnValue = 0;
    for(int ii = NDim - 1; ii >= 0; ii--)
    {
        returnValue += tmp * indicesMD(ii);
        tmp = tmp * NGrids(ii);
    }

    return returnValue;
}
