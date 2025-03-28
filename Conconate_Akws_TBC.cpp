#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include <complex>
#include <vector>
#include <math.h>
using namespace std;
#define PI_ 3.14159265

typedef vector< complex<double> >  Mat_1_Complex_doub;
typedef vector<Mat_1_Complex_doub> Mat_2_Complex_doub;
typedef vector<Mat_2_Complex_doub> Mat_3_Complex_doub;

typedef vector<double>  Mat_1_doub;
typedef vector<Mat_1_doub> Mat_2_doub;

typedef vector<string>  Mat_1_string;

struct pair_int{
    int first;
    int second;
};
typedef vector<pair_int> Mat_1_intpair;


int main(){


int NCellsx, NCellsy;
NCellsx=2;
NCellsy=2;
int Lx, Ly;
Lx=6;
Ly=6;
double dw=0.005;
double w_min=-5.0;
double w_max=3.0;
int omega_ind_max =int((w_max-w_min)/dw);

int kx_i, ky_i;
//-2.96157   0    0.0254642    0.0254642
//0.5   0   0   0   0    -2.95657



//0.5   0   0   0   0    -2.91657   9    0.0279204    0.0279204

Mat_3_Complex_doub Aup, Adn;
Aup.resize(Lx*NCellsx);
Adn.resize(Lx*NCellsx);
for(int ix=0;ix<Aup.size();ix++){
Aup[ix].resize(Ly*NCellsy);
Adn[ix].resize(Ly*NCellsy);
for(int iy=0;iy<Aup[ix].size();iy++){
Aup[ix][iy].resize(omega_ind_max);
Adn[ix][iy].resize(omega_ind_max);
}
}


double temp_doub;
for(int mx=0;mx<NCellsx;mx++){
for(int my=0;my<NCellsy;my++){


//Aq_orb0_mx0_my0.txt
string File_IN_= "Aq_orb0_mx" + to_string(mx) + "_my" + to_string(my) + ".txt";
ifstream file_in(File_IN_.c_str());
cout<<"\""<<File_IN_<<"\""<<endl;

for(int ix=0;ix<Lx;ix++){
for(int iy=0;iy<Ly;iy++){

kx_i=(NCellsx*ix)+mx;
ky_i=(NCellsy*iy)+my;

double kx,ky;
kx = (ix + ((1.0*mx)/(NCellsx)))*2.0*PI_*(1.0/Lx);
ky = (iy + ((1.0*my)/(NCellsy)))*2.0*PI_*(1.0/Ly);

double kx_temp, ky_temp;
double omega_val_temp, omega_val;
int omega_ind_temp;

for(int omega_ind=0;omega_ind<omega_ind_max;omega_ind++){

//0.5   0   0   0   0    -2.91657   9    0.0279204    0.0279204
file_in >>kx_temp>>ky_temp>>temp_doub>>temp_doub>>temp_doub>>omega_val_temp>>omega_ind_temp>>Aup[kx_i][ky_i][omega_ind]>>Adn[kx_i][ky_i][omega_ind];


assert(omega_ind_temp==omega_ind);

if(abs(kx_temp-kx)>0.00001){
cout<<"kx_temp = "<<kx_temp<<" kx="<<kx<<endl;
assert(abs(kx_temp-kx)<0.00001);
}

assert(abs(ky_temp-ky)<0.00001);

}
}
}

}
}


string File_OUT_= "Aq_orb0_TBC_" + to_string(NCellsx) + "X" + to_string(NCellsy) + "_System"+ to_string(Lx) + "X" + to_string(Ly)  +".txt";
ofstream file_out(File_OUT_.c_str());



for(int ix=0;ix<Lx;ix++){
for(int iy=0;iy<Ly;iy++){

for(int mx=0;mx<NCellsx;mx++){
for(int my=0;my<NCellsy;my++){


kx_i=(NCellsx*ix)+mx;
ky_i=(NCellsy*iy)+my;

double kx,ky;
kx = (ix + ((1.0*mx)/(NCellsx)))*2.0*PI_*(1.0/Lx);
ky = (iy + ((1.0*my)/(NCellsy)))*2.0*PI_*(1.0/Ly);

double k_index;
k_index = kx_i + ky_i*(Lx*NCellsx);


for(int omega_ind=0;omega_ind<omega_ind_max;omega_ind++){

file_out<<k_index<<"   "<<kx<<"   "<<ky<<"   "<<omega_ind<<"    "<< w_min + (omega_ind*dw) <<"   "<<Aup[kx_i][ky_i][omega_ind].real()<<"   "<<Adn[kx_i][ky_i][omega_ind].real()<<endl;
}

file_out<<endl;
}
}
}
}




//For TriangularLattice
if(Ly==Lx){
int lx_=NCellsx*Lx;
int ly_=NCellsy*Ly;
            //Create Path Gamma--> M---->K--->Gamma
            int n1, n2;
            Mat_1_intpair k_path;
            pair_int temp_pair;


            // ---k_path---------

            //--------\Gamma to M----------------
            n1=0;
            n2=0;
            while (n1<=int((lx_)/2))
            {
                temp_pair.first = n1;
                temp_pair.second = n2;
                k_path.push_back(temp_pair);
                n2++;
                n1++;
            }
            //----------------------------------

            //--------\M to K----------------
            n1=int(lx_/2)+1;
            n2=int(ly_/2)-1;
            while (n1<=int((2*lx_)/3))
            {
                temp_pair.first = n1;
                temp_pair.second = n2;
                k_path.push_back(temp_pair);
                n2--;
                n1++;
            }
            //----------------------------------

            //--------K to \Gamma----------------
            n1=int((2*lx_)/3)-2;
            n2=int((ly_)/3)-1;
            while (n1>=0)
            {
                temp_pair.first = n1;
                temp_pair.second = n2;
                k_path.push_back(temp_pair);
                n2--;
                n1--;
                n1--;
            }
            //----------------------------------

            temp_pair.first = 0;
            temp_pair.second = 0;
            k_path.push_back(temp_pair);
            temp_pair.first = 0;
            temp_pair.second = 0;
            k_path.push_back(temp_pair);

            //----------------------------------
            cout<<"PRINTING PATH"<<endl;
            for (int k_point = 0; k_point < k_path.size(); k_point++)
            {
                cout<<k_path[k_point].first<< "   "<<k_path[k_point].second<<endl;
            }


            //----k_path done-------


            string fileout_path = "Aqw_orb0_TBC" + to_string(NCellsx) + "X" + to_string(NCellsy) + "_System"+ to_string(Lx) + "X" + to_string(Ly) + "_Gamma_to_M_to_K_Path.txt";
            ofstream file_Akw_out_path(fileout_path.c_str());


			
	
            file_Akw_out_path<<"#k_point   kx     ky    omega_ind    omega_val        Akw[orb,spin=0]      Akw[orb,spin=0]"<<endl;
            for (int k_point = 0; k_point < k_path.size(); k_point++)
            {

                n1 = k_path[k_point].first;
                n2 = k_path[k_point].second;

		double kx, ky;
                kx = (2.0 * PI_ * n1) / (1.0 * lx_);
                ky = (2.0 * PI_ * n2) / (1.0 * ly_);

                for (int omega_ind = 0; omega_ind < omega_ind_max; omega_ind++)
                {
		file_Akw_out_path<<k_point<<"   "<<kx<<"   "<<ky<<"   "<<omega_ind<<"    "<< w_min + (omega_ind*dw) <<"   "<<Aup[n1][n2][omega_ind].real()<<"   "<<Adn[n1][n2][omega_ind].real()<<endl;
	}
	file_Akw_out_path<<endl;
	}
		





}



}
