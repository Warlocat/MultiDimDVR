#ifndef DVR_H_
#define DVR_H_
#include<Eigen/Dense>
using namespace Eigen;

/*
    Class for multidimensional DVR solver
    Ref: J. Chem. Phys. 96, 1982 (1992); https://doi.org/10.1063/1.462100
    
    **** Eigen3 is required ****

    Parameters:
        NDim:               Dimension number
        NStates:            Number of states included in DVR
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
    int NDim, NStates, length;
    VectorXd eigenValues;
    MatrixXd kineticT, potentialV, eigenStates;
    MatrixXi indice;
    VectorXi NGrids;
    bool solved = false, saveMem = false, readPESfromFile = false;
    MatrixXd (* PotentialPointer)(const VectorXd& Coord, const int& ND);

    /* Initialization */
    void buildDVR();

    /* Special evaluation functions used in memory saving purpose */
    void H_times_V(const VectorXd& V, VectorXd& result)const;
    
public:
    VectorXd CoordStart, CoordEnd, mass, dx;
    
    /* Construction function */
    DVR(const int& NDim_, const VectorXi& NGrids_, const VectorXd& CoordStart_, const VectorXd& CoordEnd_, const VectorXd& mass_, MatrixXd (* PotentialPointer_)(const VectorXd& Coord, const int& ND), const int& NStates_ = 1, const bool& saveMem_ = false, const bool& readPESfromFile_ = false);
	
    ~DVR();

    /* Function to return element of one-dimensional DVR kinetic energy matrix based on sinc */
    double oneDimK(const double& deltaX, const double& massX, const int& iii, const int& jjj);
    
    /* Solving the eigen-values and eigen-vectors for Hamiltonian */
    void kernel(VectorXd& energies, MatrixXd& states);
    void kernel(VectorXd& energies, MatrixXd& states, const int& solverSize, const bool& lanczos = false);

    /* Public functions to get corresponding private matrices */
    MatrixXd getHamiltonian();
    MatrixXd getEigenStates();
    VectorXd getEigenValues();

    /* Coordinate indices change */
    VectorXi oneD2mD(const int& lll)const;
    int mD2oneD(const VectorXi& indicesMD)const;
};


#endif
