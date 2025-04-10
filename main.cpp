#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <random>
#include <complex>
#include <cmath>
#include <cassert>
using namespace std;
#include "src/Matrix.h"
#include "src/ParametersEngine.h"
//#include "MFParams.h"
//#include "Hamiltonian.h"
//#include "Observables.h"
#include "src/PTMCEngine.h"
#include "random"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[]) {
    if (argc<2) { throw std::invalid_argument("USE:: executable inputfile"); }

    string ex_string_original =argv[0];
    cout<<ex_string_original<<endl;
    string ex_string;
    ex_string=ex_string_original.substr (ex_string_original.length() - 2);
    cout<<ex_string<<endl;

    string inputfile = argv[1];



    Parameters Parameters_;
    Parameters_.Initialize(inputfile);



#ifdef _OPENMP
    double begin_time, end_time;
    begin_time = omp_get_wtime();

    int N_p = omp_get_max_threads();
    omp_set_num_threads(Parameters_.NProcessors);

    cout<<"Max threads which can be used parallely = "<<N_p<<endl;
    cout<<"No. of threads used parallely = "<<Parameters_.NProcessors<<endl;
#endif


    mt19937_64 Generator_(Parameters_.RandomSeed); //for random fields

    //mt19937_64 Generator2_(Parameters_.RandomDisorderSeed); //for random disorder

    //MFParams MFParams_(Parameters_,Coordinates_,Generator_, Generator2_);

	//assert(false);

    //double wait;
   //cin>>wait;
    //Hamiltonian Hamiltonian_(Parameters_,Coordinates_,CoordinatesCluster_,MFParams_);

	//assert(false);

    //Observables Observables_(Parameters_,Coordinates_,MFParams_,Hamiltonian_);



    if (ex_string=="P2"){

     cout<<setprecision(9);
     PTMCEngine MCEngine_(Parameters_, Generator_);

     //assert(false);

     MCEngine_.Create_Connections_wrt_site();
     MCEngine_.InitializeEngine();

     MCEngine_.RUN_MC();      // Monte-Carlo Engine

    }


    else {
        cout <<"Executable not present"<<endl;
    }





    cout << "--------THE END--------" << endl;
} // main
