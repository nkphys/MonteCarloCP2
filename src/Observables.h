#include "ParametersEngine.h"
#include "Coordinates.h"
#include "MFParams.h"
#include "Hamiltonian.h"
#include "tensor_type.h"

#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#define PI acos(-1.0)

class Observables
{
public:
    Observables(Parameters &Parameters__, Coordinates &Coordinates__,
                MFParams &MFParams__, Hamiltonian &Hamiltonian__)
        : Parameters_(Parameters__), Coordinates_(Coordinates__), MFParams_(MFParams__),
          Hamiltonian_(Hamiltonian__),
          lx_(Parameters_.lx), ly_(Parameters_.ly), ncells_(Coordinates_.ncells_), nbasis_(Coordinates_.nbasis_), n_orbs_(Coordinates_.n_orbs_), n_Spins_(Parameters_.n_Spins)
    {
        Initialize();
    }

    void Initialize();

    void OccDensity();
    void Calculate_Akw();
    void Calculate_Akw_faster();
    void Calculate_Skw();
    void Calculate_Nw();
    void Get_Non_Interacting_dispersion();
    double Lorentzian(double x, double brd);
    void TotalOccDensity();
    void DensityOfStates();
    void Calculate_OrbResolved_Nw();
    void SiSjFULL();
    double fermi_function(int n);


    void calculate_quantum_SiSj();
    void quantum_SiSjQ_Average();
    void quantum_SiSj_Average();

    void SiSjQ_Average();
    void SiSj_Average();
    void Total_Energy_Average(double Curr_QuantE, double CurrE);

    void OccDensity(int tlabel);
    void DOSprint(int tlabel);
    complex<double> SiSjQ(int i, int j);
    double SiSj(int i, int j);
    double Omega(int i);

    complex<double> SiSjQ_Mean(int i, int j);
    complex<double> SiSjQ_square_Mean(int i, int j);

    double SiSj_Mean(int i, int j);
    double SiSj_square_Mean(int i, int j);

    double BandWidth;


    Matrix<complex<double>> SiSjQ_, SiSjQ_Mean_, SiSjQ_square_Mean_;
    Matrix<double> SiSj_Mean_, SiSj_square_Mean_;

    double Nematic_order_mean_, Nematic_order_square_mean_;
    Parameters &Parameters_;
    Coordinates &Coordinates_;
    MFParams &MFParams_;
    Hamiltonian &Hamiltonian_;
    int lx_, ly_, ncells_, nbasis_;
    int n_orbs_, n_Spins_;
    double dosincr_, tpi_;
    Matrix<double> SiSj_, dos;
    Mat_2_doub sx_, sy_, sz_;
    double AVG_Total_Energy, AVG_Total_Energy_sqr;

    Mat_2_doub local_density;
    Mat_2_doub local_density_Mean;
    Mat_2_doub local_density_square_Mean;

    Mat_2_Complex_doub Pauli_x, Pauli_y, Pauli_z;

    Matrix<complex<double>> quantum_SiSjQ_, quantum_SiSjQ_Mean_, quantum_SiSjQ_square_Mean_;
    Matrix<complex<double>> quantum_SiSj_, quantum_SiSj_Mean_, quantum_SiSj_square_Mean_;

    void calculate_local_density();
    void local_density_average();
};
/*
 * ***********
 *  Functions in Class Observables ------
 *  ***********
*/


void Observables::Calculate_Skw(){

    Pauli_x.resize(2);Pauli_y.resize(2);Pauli_z.resize(2);

    for(int i=0;i<2;i++){
        Pauli_x[i].resize(2); Pauli_y[i].resize(2);Pauli_z[i].resize(2);
        for(int j=0;j<2;j++){
            Pauli_x[i][j]=0;Pauli_y[i][j]=0;Pauli_z[i][j]=0;
        }
    }
    Pauli_x[0][1]=1.0;Pauli_x[1][0]=1.0;
    Pauli_y[0][1]=-1.0*iota_complex;Pauli_y[1][0]=1.0*iota_complex;
    Pauli_z[0][0]=1.0;Pauli_z[1][1]=-1.0;


    //---------Read from input file-----------------------//
    string fileout_temp = "Skw";
    double omega_min, omega_max, d_omega;
    double eta = 0.02;
    omega_min = 0;
    omega_max = 7.0;//Hamiltonian_.eigs_[Hamiltonian_.Ham_.n_row()-1] - Hamiltonian_.eigs_[0] +1.0;
    d_omega = 0.005;
    //---------------------------------------------------//


    int mx = Parameters_.TBC_mx;
    int my = Parameters_.TBC_my;

    int omega_index_max = int((omega_max - omega_min) / (d_omega));



    for(int orb1=0;orb1<n_orbs_;orb1++){
        for(int orb2=orb1;orb2<n_orbs_;orb2++){


            int c1, c2, c3, c4;
            int lp, lp_x, lp_y;
            int r_x, r_y, l_x, l_y;

            Mat_3_Complex_doub B_mat;
            B_mat.resize(ncells_);
            for(int i=0;i<ncells_;i++){
                B_mat[i].resize(Hamiltonian_.Ham_.n_row());
                for(int j=0;j<Hamiltonian_.Ham_.n_row();j++){
                    B_mat[i][j].resize(Hamiltonian_.Ham_.n_row());
                }
            }


            for(int r=0;r<ncells_;r++){
                r_x=Coordinates_.indx_cellwise(r);
                r_y=Coordinates_.indy_cellwise(r);

                for(int lambda=0;lambda<Hamiltonian_.Ham_.n_row();lambda++){
                    for(int lambda_p=0;lambda_p<Hamiltonian_.Ham_.n_row();lambda_p++){
                        B_mat[r][lambda][lambda_p]=zero_complex;

                        for(int alpha=0;alpha<2;alpha++){
                            for(int alpha_p=0;alpha_p<2;alpha_p++){

                                for(int beta=0;beta<2;beta++){
                                    for(int beta_p=0;beta_p<2;beta_p++){

                                        for(int l=0;l<ncells_;l++){
                                            l_x=Coordinates_.indx_cellwise(l);
                                            l_y=Coordinates_.indy_cellwise(l);

                                            lp_x = (l_x + r_x)%lx_;
                                            lp_y = (l_y + r_y)%ly_;

                                            lp = Coordinates_.Ncell(lp_x, lp_y);

                                            c1=Coordinates_.Nbasis(l_x, l_y, orb1) + Coordinates_.nbasis_*alpha;
                                            c2=Coordinates_.Nbasis(l_x, l_y, orb1) + Coordinates_.nbasis_*beta;
                                            c3=Coordinates_.Nbasis(lp_x, lp_y, orb2) + Coordinates_.nbasis_*alpha_p;
                                            c4=Coordinates_.Nbasis(lp_x, lp_y, orb2) + Coordinates_.nbasis_*beta_p;

                                            B_mat[r][lambda][lambda_p]+= conj(Hamiltonian_.Ham_(c1, lambda)) * Hamiltonian_.Ham_(c2, lambda_p)*
                                                    conj(Hamiltonian_.Ham_(c3, lambda_p)) * Hamiltonian_.Ham_(c4, lambda)*
                                                    ( (Pauli_x[alpha][beta]*Pauli_x[alpha_p][beta_p])
                                                      + (Pauli_y[alpha][beta]*Pauli_y[alpha_p][beta_p])
                                                      + (Pauli_z[alpha][beta]*Pauli_z[alpha_p][beta_p])
                                                      );

                                        }

                                    }
                                }

                            }
                        }
                    }
                }
            }




            Mat_2_Complex_doub A_mat;
            A_mat.resize(ncells_);
            for(int r=0;r<ncells_;r++){
                A_mat[r].resize(omega_index_max);
            }


            for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
            {
                for(int r=0;r<ncells_;r++){
                    r_x=Coordinates_.indx_cellwise(r);
                    r_y=Coordinates_.indy_cellwise(r);

                    for(int lambda=0;lambda<Hamiltonian_.Ham_.n_row();lambda++){
                        for(int lambda_p=0;lambda_p<Hamiltonian_.Ham_.n_row();lambda_p++){

                            if(lambda != lambda_p){
                                A_mat[r][omega_ind] += B_mat[r][lambda][lambda_p]*
                                        (1.0/(1.0 + exp(Parameters_.beta*(Hamiltonian_.eigs_[lambda] - Parameters_.mus))))*
                                        (1.0/(1.0 + exp(-1.0*Parameters_.beta*(Hamiltonian_.eigs_[lambda_p] - Parameters_.mus))))*
                                        Lorentzian(omega_min + (omega_ind * d_omega) + Hamiltonian_.eigs_[lambda] - Hamiltonian_.eigs_[lambda_p], eta);
                            }
                        }
                    }
                }
            }



            complex<double> temp_Skw;
            double kx, ky;
            int kx_i, ky_i;

            int n1, n2;
            Mat_1_intpair k_path, k_path2;
            pair_int temp_pair;

            for(int n1_=0;n1_<lx_;n1_++){
                for(int n2_=0;n2_<ly_;n2_++){
                    temp_pair.first = n1_;
                    temp_pair.second = n2_;
                    k_path2.push_back(temp_pair);
                }
            }

            //Run assuming Kagome
            if((ly_==lx_)) { //For triangular lattices (example Kagome)

                assert(lx_==ly_);
                //Create Path Gamma--> M---->K--->Gamma
                // ---k_path---------

                //--------\Gamma to M----------------
                n1=0;
                n2=0;
                while (n1<=int(lx_/2))
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

            }



            if(ly_==1) { //For 1d lattice

                assert(ly_==1);
                //Create Path 0 to 2pi
                // ---k_path---------


                n1=0;
                n2=0;
                while (n1<=int(lx_))
                {
                    temp_pair.first = n1;
                    temp_pair.second = n2;
                    k_path.push_back(temp_pair);
                    n1++;
                }
                //----------------------------------


                //----------------------------------

                temp_pair.first = 0;
                temp_pair.second = 0;
                k_path.push_back(temp_pair);
                temp_pair.first = 0;
                temp_pair.second = 0;
                k_path.push_back(temp_pair);

            }


            //----------------------------------
            cout<<"PRINTING PATH"<<endl;
            for (int k_point = 0; k_point < k_path.size(); k_point++)
            {
                cout<<k_path[k_point].first<< "   "<<k_path[k_point].second<<endl;
            }


            //----k_path done-------


            int r_posx, r_posy;
            string fileout_path = fileout_temp + "_orbs" + to_string(orb1)+ "_"+ to_string(orb2) + "_for_Full.txt";
            ofstream file_Skw_out_path(fileout_path.c_str());


            file_Skw_out_path<<"#k_point   n1     n2    omega_val    omega_ind       Skw[orb1, orb2]"<<endl;
            for (int k_point = 0; k_point < k_path2.size(); k_point++)
            {

                n1 = k_path2[k_point].first;
                n2 = k_path2[k_point].second;

                kx = (2.0 * PI * n1) / (1.0 * Parameters_.lx);
                ky = (2.0 * PI * n2) / (1.0 * Parameters_.ly);

                for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
                {
                    temp_Skw = zero_complex;

                    for (int r = 0; r < Coordinates_.ncells_; r++)
                    {
                        r_posx = Coordinates_.indx_cellwise(r);
                        r_posy = Coordinates_.indy_cellwise(r);

                        temp_Skw += one_complex *
                                exp(iota_complex * (kx * (r_posx) +
                                                    ky * (r_posy) )) *
                                A_mat[r][omega_ind];
                    }

                    //Use 1:4:6(7)----for gnuplot

                    //kx+(((1.0 * mx)/ (1.0 * Parameters_.TBC_cellsX))*(2.0*PI/Parameters_.lx))<<"   "<<ky+(((1.0 * my)/ (1.0 * Parameters_.TBC_cellsY))*(2.0*PI/Parameters_.ly))<<"   "<<kx_i << "   " << ky_i << "   " << (ky_i * Parameters_.lx) + kx_i << "    " << omega_min + (d_omega * omega_ind) << "   " << omega_ind << "    "
                    //<< temp_up.real() << "    " << temp_dn.real() << "
                    file_Skw_out_path << kx+(((1.0 * mx)/ (1.0 * Parameters_.TBC_cellsX))*(2.0*PI/Parameters_.lx))<<"   "<<ky+(((1.0 * my)/ (1.0 * Parameters_.TBC_cellsY))*(2.0*PI/Parameters_.ly))
                                      << "    " <<n1<<"    "<<n2<< "    "<<(n2 * Parameters_.lx) + n1 <<"   "<<omega_min + (d_omega * omega_ind) << "   " << omega_ind << "    "
                                     << temp_Skw.real()<<"   "<<temp_Skw.imag()
                                     << endl;
                }
                file_Skw_out_path << endl;

            }









        }
    }




}


void Observables::Calculate_Akw_faster()
{

    //NEED TO CHANGE

    //---------Read from input file-----------------------//
    string fileout_temp = "Akw";
    double omega_min, omega_max, d_omega;
    double eta = 0.01;
    omega_min = -5.0;//Hamiltonian_.eigs_[0] -1.0;
    omega_max = 3.0;//Hamiltonian_.eigs_[Hamiltonian_.Ham_.n_row()-1] +1.0;
    d_omega = 0.005;
    //---------------------------------------------------//

    int omega_index_max = int((omega_max - omega_min) / (d_omega));



    int c1, c2;
    int j_posx, j_posy, r_posy;
    int l_posx, l_posy, r_posx;

    Mat_3_Complex_doub Psi_amp;
    Psi_amp.resize(Hamiltonian_.Ham_.n_row());
    for(int i=0;i<Hamiltonian_.Ham_.n_row();i++){
        Psi_amp[i].resize(Coordinates_.ncells_);
        for(int j=0;j<Coordinates_.ncells_;j++){
            Psi_amp[i][j].resize(n_orbs_*2); //  orb + n_orbs_*spin
        }
    }



    int dis_x, dis_y, dis_r;

    for (int n = 0; n < Hamiltonian_.Ham_.n_row(); n++)
    {
        for (int j = 0; j < Coordinates_.ncells_; j++)
        {
            j_posx = Coordinates_.indx_cellwise(j);
            j_posy = Coordinates_.indy_cellwise(j);
            for (int l = 0; l < Coordinates_.ncells_; l++)
            {
                l_posx = Coordinates_.indx_cellwise(l);
                l_posy = Coordinates_.indy_cellwise(l);
                dis_x = (j_posx - l_posx + Coordinates_.lx_)%lx_;
                dis_y = (j_posy - l_posy + Coordinates_.ly_)%ly_;
                dis_r=Coordinates_.Ncell(dis_x, dis_y);

                for(int orb=0;orb<n_orbs_;orb++){
                    for(int spin=0;spin<2;spin++){
                        c1 = Coordinates_.Nbasis(l_posx, l_posy, orb) + Coordinates_.nbasis_*spin;
                        c2 = Coordinates_.Nbasis(j_posx, j_posy, orb) + Coordinates_.nbasis_*spin;
                        Psi_amp[n][dis_r][orb + n_orbs_*spin] += conj(Hamiltonian_.Ham_(c1, n)) * Hamiltonian_.Ham_(c2, n);
                    }
                }
            }
        }
    }



    for(int orb=0;orb<Parameters_.n_orbs;orb++){

        //cout<<"here "<<orb<<endl;
        string fileout = fileout_temp + "_orb" + to_string(orb) + ".txt";
        ofstream file_Akw_out(fileout.c_str());
        Mat_2_Complex_doub A_up, A_dn;
        A_up.resize(Coordinates_.ncells_);
        A_dn.resize(Coordinates_.ncells_);
        for (int i = 0; i < Coordinates_.ncells_; i++)
        {
            A_up[i].resize(omega_index_max);
            A_dn[i].resize(omega_index_max);
        }


        complex<double> Nup_check(0, 0);
        complex<double> Ndn_check(0, 0);

        for (int r = 0; r < Coordinates_.ncells_; r++)
        {

            for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
            {
                A_up[r][omega_ind] = zero_complex;
                A_dn[r][omega_ind] = zero_complex;

                for (int n = 0; n < Hamiltonian_.Ham_.n_row(); n++)
                {
                    //c= l + or1*ns_ + ns_*orbs_*spin;
                    //Hamiltonian_.Ham_(c2,n) is nth eigenvector and c2th component [checked];
                    //                        c1 = Coordinates_.Nbasis(l_posx, l_posy, orb) + Coordinates_.nbasis_;
                    //                        c2 = Coordinates_.Nbasis(j_posx, j_posy, orb) + Coordinates_.nbasis_;
                    A_dn[r][omega_ind] += Psi_amp[n][r][orb + n_orbs_*1] *
                            Lorentzian(omega_min + (omega_ind * d_omega) - Hamiltonian_.eigs_[n], eta);

                    //                        c1 = Coordinates_.Nbasis(l_posx, l_posy, orb);
                    //                        c2 = Coordinates_.Nbasis(j_posx, j_posy, orb);
                    A_up[r][omega_ind] += Psi_amp[n][r][orb + n_orbs_*0] *
                            Lorentzian(omega_min + (omega_ind * d_omega) - Hamiltonian_.eigs_[n], eta);

                }

                if (r==0)
                {
                    Nup_check += (A_up[r][omega_ind]) * d_omega;
                    Ndn_check += (A_dn[r][omega_ind]) * d_omega;
                }
            }
        }



        cout << "Nup_check = " << Nup_check << endl;
        cout << "Ndn_check = " << Ndn_check << endl;



        complex<double> temp_up;
        complex<double> temp_dn;
        double kx, ky;
        int kx_i, ky_i;
        int mx = Parameters_.TBC_mx;
        int my = Parameters_.TBC_my;


        for (int kx_i_ = 0; kx_i_ <= Parameters_.lx; kx_i_++)
        {
            for (int ky_i_ = 0; ky_i_ < Parameters_.ly; ky_i_++)
            {

                kx_i = kx_i_;
                ky_i = ky_i_;
                //kx = ((2.0 * PI * (1.0*kx_i  +  Parameters_.BoundaryConnection*((1.0 * mx)/ (1.0 * Parameters_.TBC_cellsX)) ) ) / (1.0 * Parameters_.lx));
                //Parameters_.BoundaryConnection*(2.0 * (1.0 * mx) * PI / (1.0 * Parameters_.TBC_cellsX));

                kx = ((2.0 * PI * (1.0*kx_i ) ) / (1.0 * Parameters_.lx));
                ky = (2.0 * PI * (1.0*ky_i )) / (1.0 * Parameters_.ly);

                for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
                {
                    temp_up = zero_complex;
                    temp_dn = zero_complex;

                    for (int r = 0; r < Coordinates_.ncells_; r++)
                    {
                        r_posx = Coordinates_.indx_cellwise(r);
                        r_posy = Coordinates_.indy_cellwise(r);

                        temp_up += one_complex *
                                exp(iota_complex * (kx * (r_posx) +
                                                    ky * (r_posy) )) *
                                A_up[r][omega_ind];

                        temp_dn += one_complex *
                                exp(iota_complex * (kx * (r_posx) +
                                                    ky * (r_posy))) *
                                A_dn[r][omega_ind];

                    }

                    //Use 3:4:6(7)----for gnuplot
                    file_Akw_out <<kx+(((1.0 * mx)/ (1.0 * Parameters_.TBC_cellsX))*(2.0*PI/Parameters_.lx))<<"   "<<ky+(((1.0 * my)/ (1.0 * Parameters_.TBC_cellsY))*(2.0*PI/Parameters_.ly))<<"   "<<kx_i << "   " << ky_i << "   " << (ky_i * Parameters_.lx) + kx_i << "    " << omega_min + (d_omega * omega_ind) << "   " << omega_ind << "    "
                                << temp_up.real() << "    " << temp_dn.real() << "    "
                                << endl;
                }
                file_Akw_out << endl;
            }
        }





        //Run assuming Kagome
        if(ly_==lx_) { //For triangular lattices (example Kagome)

            assert(lx_==ly_);
            //Create Path Gamma--> M---->K--->Gamma
            int n1, n2;
            Mat_1_intpair k_path;
            pair_int temp_pair;


            // ---k_path---------

            //--------\Gamma to M----------------
            n1=0;
            n2=0;
            while (n1<=int(lx_/2))
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


            string fileout_path = fileout_temp + "_orb" + to_string(orb) + "_Gamma_to_M_to_K_Path.txt";
            ofstream file_Akw_out_path(fileout_path.c_str());


            file_Akw_out_path<<"#k_point   n1     n2    omega_val    omega_ind        Akw[orb,spin=0]      Akw[orb,spin=0]"<<endl;
            for (int k_point = 0; k_point < k_path.size(); k_point++)
            {

                n1 = k_path[k_point].first;
                n2 = k_path[k_point].second;

                kx = (2.0 * PI * n1) / (1.0 * Parameters_.lx);
                ky = (2.0 * PI * n2) / (1.0 * Parameters_.ly);

                for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
                {
                    temp_up = zero_complex;
                    temp_dn = zero_complex;

                    for (int r = 0; r < Coordinates_.ncells_; r++)
                    {
                        r_posx = Coordinates_.indx_cellwise(r);
                        r_posy = Coordinates_.indy_cellwise(r);

                        temp_up += one_complex *
                                exp(iota_complex * (kx * (r_posx) +
                                                    ky * (r_posy) )) *
                                A_up[r][omega_ind];

                        temp_dn += one_complex *
                                exp(iota_complex * (kx * (r_posx) +
                                                    ky * (r_posy))) *
                                A_dn[r][omega_ind];

                    }

                    //Use 1:4:6(7)----for gnuplot
                    file_Akw_out_path <<k_point<<"   "<<kx+(((1.0 * mx)/ (1.0 * Parameters_.TBC_cellsX))*(2.0*PI/Parameters_.lx)) << "   " << ky+(((1.0 * my)/ (1.0 * Parameters_.TBC_cellsY))*(2.0*PI/Parameters_.ly))  << "    " << omega_min + (d_omega * omega_ind) << "   " << omega_ind << "    "
                                     << temp_up.real() << "    " << temp_dn.real() << "    "
                                     << endl;
                }
                file_Akw_out_path << endl;

            }

        }



    }



}


void Observables::Calculate_Akw()
{

    //NEED TO CHANGE

    //---------Read from input file-----------------------//
    string fileout_temp = "Akw";
    double omega_min, omega_max, d_omega;
    double eta = 0.01;
    omega_min = Hamiltonian_.eigs_[0] -1.0;
    omega_max = Hamiltonian_.eigs_[Hamiltonian_.Ham_.n_row()-1] +1.0;
    d_omega = 0.005;
    //---------------------------------------------------//

    int omega_index_max = int((omega_max - omega_min) / (d_omega));



    int c1, c2;
    int j_posx, j_posy;
    int l_posx, l_posy;


    for(int orb=0;orb<Parameters_.n_orbs;orb++){

        //cout<<"here "<<orb<<endl;
        string fileout = fileout_temp + "_orb" + to_string(orb) + ".txt";
        ofstream file_Akw_out(fileout.c_str());
        Mat_3_Complex_doub A_up, A_dn;
        A_up.resize(Coordinates_.ncells_);
        A_dn.resize(Coordinates_.ncells_);
        for (int i = 0; i < Coordinates_.ncells_; i++)
        {
            A_up[i].resize(Coordinates_.ncells_);
            A_dn[i].resize(Coordinates_.ncells_);

            for (int j = 0; j < Coordinates_.ncells_; j++)
            {
                A_up[i][j].resize(omega_index_max);
                A_dn[i][j].resize(omega_index_max);
            }
        }


        complex<double> Nup_check(0, 0);
        complex<double> Ndn_check(0, 0);

        for (int j = 0; j < Coordinates_.ncells_; j++)
        {
            j_posx = Coordinates_.indx_cellwise(j);
            j_posy = Coordinates_.indy_cellwise(j);

            for (int l = 0; l < Coordinates_.ncells_; l++)
            {
                l_posx = Coordinates_.indx_cellwise(l);
                l_posy = Coordinates_.indy_cellwise(l);

                for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
                {
                    A_up[j][l][omega_ind] = zero_complex;
                    A_dn[j][l][omega_ind] = zero_complex;


                    for (int n = 0; n < Hamiltonian_.Ham_.n_row(); n++)
                    {

                        //c= l + or1*ns_ + ns_*orbs_*spin;
                        //Hamiltonian_.Ham_(c2,n) is nth eigenvector and c2th component [checked];
                        c1 = Coordinates_.Nbasis(l_posx, l_posy, orb) + Coordinates_.nbasis_;
                        c2 = Coordinates_.Nbasis(j_posx, j_posy, orb) + Coordinates_.nbasis_;
                        A_dn[j][l][omega_ind] += conj(Hamiltonian_.Ham_(c1, n)) * Hamiltonian_.Ham_(c2, n) *
                                Lorentzian(omega_min + (omega_ind * d_omega) - Hamiltonian_.eigs_[n], eta);

                        c1 = Coordinates_.Nbasis(l_posx, l_posy, orb);
                        c2 = Coordinates_.Nbasis(j_posx, j_posy, orb);
                        A_up[j][l][omega_ind] += conj(Hamiltonian_.Ham_(c1, n)) * Hamiltonian_.Ham_(c2, n) *
                                Lorentzian(omega_min + (omega_ind * d_omega) - Hamiltonian_.eigs_[n], eta);

                    }

                    if (j == l)
                    {
                        Nup_check += (A_up[j][l][omega_ind]) * d_omega;
                        Ndn_check += (A_dn[j][l][omega_ind]) * d_omega;
                    }
                }
            }
        }


        cout << "Nup_check = " << Nup_check << endl;
        cout << "Ndn_check = " << Ndn_check << endl;



        complex<double> temp_up;
        complex<double> temp_dn;
        double kx, ky;
        int kx_i, ky_i;


        for (int kx_i_ = 0; kx_i_ <= Parameters_.lx; kx_i_++)
        {
            for (int ky_i_ = 0; ky_i_ < Parameters_.ly; ky_i_++)
            {

                kx_i = kx_i_;
                ky_i = ky_i_;
                kx = (2.0 * PI * kx_i) / (1.0 * Parameters_.lx);
                ky = (2.0 * PI * ky_i) / (1.0 * Parameters_.ly);

                for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
                {
                    temp_up = zero_complex;
                    temp_dn = zero_complex;

                    for (int j = 0; j < Coordinates_.ncells_; j++)
                    {
                        j_posx = Coordinates_.indx_cellwise(j);
                        j_posy = Coordinates_.indy_cellwise(j);

                        for (int l = 0; l < Coordinates_.ncells_; l++)
                        {
                            l_posx = Coordinates_.indx_cellwise(l);
                            l_posy = Coordinates_.indy_cellwise(l);

                            temp_up += one_complex *
                                    exp(iota_complex * (kx * (j_posx - l_posx) +
                                                        ky * (j_posy - l_posy) )) *
                                    A_up[j][l][omega_ind];

                            temp_dn += one_complex *
                                    exp(iota_complex * (kx * (j_posx - l_posx) +
                                                        ky * (j_posy - l_posy))) *
                                    A_dn[j][l][omega_ind];

                        }
                    }
                    //Use 3:4:6(7)----for gnuplot
                    file_Akw_out <<kx_i << "   " << ky_i << "   " << (ky_i * Parameters_.lx) + kx_i << "    " << omega_min + (d_omega * omega_ind) << "   " << omega_ind << "    "
                                << temp_up.real() << "    " << temp_dn.real() << "    "
                                << endl;
                }
                file_Akw_out << endl;
            }
        }



    }




}

void Observables::Calculate_Nw()
{


    //---------Read from input file-----------------------//
    string fileout = "Nw_total.txt";
    double omega_min, omega_max, d_omega;
    double eta = 0.1;
    omega_min = Hamiltonian_.eigs_[0] -1.0;
    omega_max = Hamiltonian_.eigs_[Hamiltonian_.Ham_.n_row()-1] +1.0;
    d_omega = 0.01;
    //---------------------------------------------------//

    int omega_index_max = int((omega_max - omega_min) / (d_omega));

    ofstream file_Nw_out(fileout.c_str());

    double temp_val;

    for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
    {

        temp_val = 0.0;

        for (int n = 0; n < Hamiltonian_.Ham_.n_row(); n++)
        {

            temp_val += Lorentzian(omega_min + (omega_ind * d_omega) - Hamiltonian_.eigs_[n], eta);
        }

        file_Nw_out << omega_min + (omega_ind * d_omega) << "     " << temp_val << "     " << endl;
    }

    file_Nw_out << "#mu = " << Parameters_.mus << endl;


}

void Observables::Calculate_OrbResolved_Nw()
{

    //NEED TO CHANGE

    //---------Read from input file-----------------------//
    string fileout = "NwOrbResolved_total.txt";
    double omega_min, omega_max, d_omega;
    double eta = 0.1;
    omega_min = -100;
    omega_max = 100.0;
    d_omega = 0.001;
    //---------------------------------------------------//

    int omega_index_max = int((omega_max - omega_min) / (d_omega));

    ofstream file_Nw_out(fileout.c_str());

    double temp_val;

    for (int omega_ind = 0; omega_ind < omega_index_max; omega_ind++)
    {

        temp_val = 0.0;

        for (int n = 0; n < Hamiltonian_.Ham_.n_row(); n++)
        {

            temp_val += Lorentzian(omega_min + (omega_ind * d_omega) - Hamiltonian_.eigs_[n], eta);
        }

        file_Nw_out << omega_min + (omega_ind * d_omega) << "     " << temp_val << "     " << endl;
    }

    file_Nw_out << "#mu = " << Parameters_.mus << endl;


}

void Observables::Get_Non_Interacting_dispersion()
{
}

double Observables::Lorentzian(double x, double brd)
{
    double temp;

    temp = (1.0 / PI) * ((brd / 2.0) / ((x * x) + ((brd * brd) / 4.0)));

    return temp;
}

void Observables::DensityOfStates()
{
    //-----------Calculate Bandwidth------//
    BandWidth = 2.0;
    //-----------------------------------//

} // ----------

void Observables::OccDensity()
{

} // ----------

void Observables::TotalOccDensity()
{

} // ----------

complex<double> Observables::SiSjQ(int i, int j) { return SiSjQ_(i, j); }

double Observables::SiSj(int i, int j) { return SiSj_(i, j); }

complex<double> Observables::SiSjQ_Mean(int i, int j) { return SiSjQ_Mean_(i, j); }

complex<double> Observables::SiSjQ_square_Mean(int i, int j) { return SiSjQ_square_Mean_(i, j); }

double Observables::SiSj_Mean(int i, int j) { return SiSj_Mean_(i, j); }

double Observables::SiSj_square_Mean(int i, int j) { return SiSj_square_Mean_(i, j); }

double Observables::fermi_function(int n)
{
    double value;
    value = 1.0 / (exp(Parameters_.beta * (Hamiltonian_.eigs_[n] - Parameters_.mus)) + 1.0);
    return value;
}

void Observables::calculate_quantum_SiSj()
{
    /*
    Matrix<complex<double>> F_u_u;
    Matrix<complex<double>> F_d_d;
    Matrix<complex<double>> F_u_d;
    Matrix<complex<double>> F_d_u;
    Matrix<complex<double>> omF_u_u;
    Matrix<complex<double>> omF_d_d;
    Matrix<complex<double>> omF_u_d;
    Matrix<complex<double>> omF_d_u;
    int nx, ny;
    int jx, jy;
    F_u_u.resize(ns_, ns_);
    F_d_d.resize(ns_, ns_);
    F_u_d.resize(ns_, ns_);
    F_d_u.resize(ns_, ns_);
    omF_u_u.resize(ns_, ns_);
    omF_d_d.resize(ns_, ns_);
    omF_u_d.resize(ns_, ns_);
    omF_d_u.resize(ns_, ns_);
    for (int i = 0; i < ns_; i++)
    {
        for (int j = 0; j < ns_; j++)
        {
            for (int n = 0; n < Hamiltonian_.eigs_.size(); n++)
            {
                F_u_u(i, j) += (conj(Hamiltonian_.Ham_(i, n)) * Hamiltonian_.Ham_(j, n) * fermi_function(n));
                F_d_d(i, j) += (conj(Hamiltonian_.Ham_(i + ns_, n)) * Hamiltonian_.Ham_(j + ns_, n) * fermi_function(n));
                F_u_d(i, j) += (conj(Hamiltonian_.Ham_(i, n)) * Hamiltonian_.Ham_(j + ns_, n) * fermi_function(n));
                F_d_u(i, j) += (conj(Hamiltonian_.Ham_(i + ns_, n)) * Hamiltonian_.Ham_(j, n) * fermi_function(n));
                omF_u_u(i, j) += (conj(Hamiltonian_.Ham_(i, n)) * Hamiltonian_.Ham_(j, n) * (1.0 - fermi_function(n)));
                omF_d_d(i, j) += (conj(Hamiltonian_.Ham_(i + ns_, n)) * Hamiltonian_.Ham_(j + ns_, n) * (1.0 - fermi_function(n)));
                omF_u_d(i, j) += (conj(Hamiltonian_.Ham_(i, n)) * Hamiltonian_.Ham_(j + ns_, n) * (1.0 - fermi_function(n)));
                omF_d_u(i, j) += (conj(Hamiltonian_.Ham_(i + ns_, n)) * Hamiltonian_.Ham_(j, n) * (1.0 - fermi_function(n)));
            }
        }
    }

    int i_;
    for (int ix = 0; ix < lx_; ix++)
    {
        for (int iy = 0; iy < ly_; iy++)
        {
            // i = Coordinates_.Nc(ix, iy);

            quantum_SiSj_(ix, iy) = 0.0;
            for (int j = 0; j < ns_; j++)
            {
                jx = Coordinates_.indx(j);
                jy = Coordinates_.indy(j);
                nx = (jx + ix) % lx_;
                ny = (jy + iy) % ly_;
                i_ = Coordinates_.Nc(nx, ny);
                quantum_SiSj_(ix, iy) += (
                0.25*(F_u_u(i_, i_) * F_u_u(j, j) + F_u_u(i_, j) * omF_u_u(j, i_)
                    - ( F_u_u(i_, i_) * F_d_d(j, j) + F_u_d(i_, j) * omF_d_u(j, i_) )
                    - ( F_d_d(i_, i_) * F_u_u(j, j) + F_d_u(i_, j) * omF_u_d(j, i_) )
                    + F_d_d(i_, i_) * F_d_d(j, j) + F_d_d(i_, j) * omF_d_d(j, i_))
                    + 0.5 * (F_u_d(i_, i_) * F_d_u(j, j) + F_u_u(i_, j) * omF_d_d(j, i_))
                    + 0.5 * (F_d_u(i_, i_) * F_u_d(j, j) + F_d_d(i_, j) * omF_u_u(j, i_))
                    ).real();
            }
            quantum_SiSj_(ix, iy) /= (ns_ * 1.0);
        }
    }

    //Fourier transform
    double phase, Cos_ij, Sin_ij;
    for (int qx = 0; qx < lx_; qx++)
    {
        for (int qy = 0; qy < ly_; qy++)
        {
            quantum_SiSjQ_(qx, qy) = complex<double>(0.0, 0.0);
            for (int xr = 0; xr < lx_; xr++)
            {
                for (int yr = 0; yr < ly_; yr++)
                {
                    phase = 2.0 * Parameters_.pi * (double(qx * xr) / double(lx_) + double(qy * yr) / double(ly_));
                    Cos_ij = cos(phase);
                    Sin_ij = sin(phase);
                    quantum_SiSjQ_(qx, qy) += quantum_SiSj_(xr, yr) * complex<double>(Cos_ij, Sin_ij);
                }
            }
            quantum_SiSjQ_(qx, qy) *= double(1.0 / (lx_ * ly_));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

    */
}

void Observables::calculate_local_density()
{
    int c1;
    complex<double> value = zero_complex;
    // cout <<"Parameter mus="<< Parameters_.mus<<endl;
    for (int i = 0; i < nbasis_; i++)
    {
        for (int sigma = 0; sigma < 2; sigma++)
        {
            local_density[i][sigma] = 0.0;
            c1 = i + (sigma * nbasis_);
            for (int n = 0; n < Hamiltonian_.eigs_.size(); n++)
            {
                local_density[i][sigma] += (conj(Hamiltonian_.Ham_(c1, n)) * Hamiltonian_.Ham_(c1, n) * fermi_function(n)).real();
            }

            // value += (conj(Hamiltonian_.Ham_(c1, 1)) * Hamiltonian_.Ham_(c1, 1));
        }
    }
}

void Observables::SiSjFULL()
{

    double Cos_ij, Sin_ij, ei, ai, phase;
    int site_, site_p, ax, ay;

    for (int i = 0; i < lx_; i++)
    {
        for (int j = 0; j < ly_; j++)
        {
            for(int spin1=0;spin1<n_Spins_;spin1++){

                site_ = Coordinates_.Ncell(i, j);
                ei = MFParams_.etheta[spin1](i, j);
                ai = MFParams_.ephi[spin1](i, j);
                sx_[spin1][site_] = MFParams_.Moment_Size[spin1](i, j)*cos(ai) * sin(ei);
                sy_[spin1][site_] = MFParams_.Moment_Size[spin1](i, j)*sin(ai) * sin(ei);
                sz_[spin1][site_] = MFParams_.Moment_Size[spin1](i, j)*cos(ei);

            }
        }
    }

    for (int xr = 0; xr < lx_; xr++)
    {
        for (int yr = 0; yr < ly_; yr++)
        {
            SiSj_(xr, yr) = double(0.0);
            for (int i = 0; i < lx_; i++)
            {
                for (int j = 0; j < ly_; j++)
                {
                    for(int spin1=0;spin1<n_Spins_;spin1++){
                        for(int spin2=0;spin2<n_Spins_;spin2++){

                            site_ = Coordinates_.Ncell(i, j);
                            ax = (i + xr) % lx_;
                            ay = (j + yr) % ly_;
                            site_p = Coordinates_.Ncell(ax, ay);
                            SiSj_(xr, yr) += sx_[spin1][site_] * sx_[spin2][site_p];
                            SiSj_(xr, yr) += sy_[spin1][site_] * sy_[spin2][site_p];
                            SiSj_(xr, yr) += sz_[spin1][site_] * sz_[spin2][site_p];
                        }
                    }
                }
            }
            SiSj_(xr, yr) *= double(1.0 / (lx_ * ly_));
            //cout << xr << " "<< yr<< " "<<  SiSj_(xr,yr) << endl;
        }
    }

    for (int qx = 0; qx < lx_; qx++)
    {
        for (int qy = 0; qy < ly_; qy++)
        {
            SiSjQ_(qx, qy) = complex<double>(0.0, 0.0);
            for (int xr = 0; xr < lx_; xr++)
            {
                for (int yr = 0; yr < ly_; yr++)
                {
                    phase = 2.0 * Parameters_.pi * (double(qx * xr) / double(lx_) + double(qy * yr) / double(ly_));
                    Cos_ij = cos(phase);
                    Sin_ij = sin(phase);
                    SiSjQ_(qx, qy) += SiSj_(xr, yr) * complex<double>(Cos_ij, Sin_ij);
                }
            }
            SiSjQ_(qx, qy) *= double(1.0 / (lx_ * ly_));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------

void Observables::SiSjQ_Average()
{

    for (int qx = 0; qx < lx_; qx++)
    {
        for (int qy = 0; qy < ly_; qy++)
        {
            SiSjQ_Mean_(qx, qy) += SiSjQ_(qx, qy);
            SiSjQ_square_Mean_(qx, qy) += (SiSjQ_(qx, qy) * SiSjQ_(qx, qy));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------

void Observables::quantum_SiSjQ_Average()
{

    for (int qx = 0; qx < lx_; qx++)
    {
        for (int qy = 0; qy < ly_; qy++)
        {
            quantum_SiSjQ_Mean_(qx, qy) += quantum_SiSjQ_(qx, qy);
            quantum_SiSjQ_square_Mean_(qx, qy) += (quantum_SiSjQ_(qx, qy) * quantum_SiSjQ_(qx, qy));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }
}

void Observables::SiSj_Average()
{

    for (int x = 0; x < lx_; x++)
    {
        for (int y = 0; y < ly_; y++)
        {
            SiSj_Mean_(x, y) += SiSj_(x, y);
            SiSj_square_Mean_(x, y) += (SiSj_(x, y) * SiSj_(x, y));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

    Nematic_order_mean_ += fabs(SiSj_(1, 0) - SiSj_(0, 1)) * 0.5;
    Nematic_order_square_mean_ += (SiSj_(1, 0) - SiSj_(0, 1)) * (SiSj_(1, 0) - SiSj_(0, 1)) * 0.25;

} // ----------

void Observables::quantum_SiSj_Average()
{

    for (int x = 0; x < lx_; x++)
    {
        for (int y = 0; y < ly_; y++)
        {
            quantum_SiSj_Mean_(x, y) += quantum_SiSj_(x, y);
            quantum_SiSj_square_Mean_(x, y) += (quantum_SiSj_(x, y) * quantum_SiSj_(x, y));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------

void Observables::local_density_average()
{
    for (int i = 0; i < nbasis_; i++)
    {
        for (int sigma = 0; sigma < 2; sigma++)
        {
            local_density_Mean[i][sigma] += local_density[i][sigma];
            local_density_square_Mean[i][sigma] += pow(local_density[i][sigma], 2);
        }
    }
}

void Observables::Total_Energy_Average(double Curr_QuantE, double CurrE)
{

    AVG_Total_Energy += Curr_QuantE + CurrE;
    AVG_Total_Energy_sqr += (Curr_QuantE + CurrE) * (Curr_QuantE + CurrE);
}

void Observables::OccDensity(int tlabel)
{

} // ----------

void Observables::DOSprint(int tlabel)
{

} // ----------

void Observables::Initialize()
{

    complex<double> zero(0.0, 0.0);
    int space = 2 * ncells_ * n_orbs_;

    sx_.resize(n_Spins_);
    sy_.resize(n_Spins_);
    sz_.resize(n_Spins_);

    for(int spin_no=0;spin_no<n_Spins_;spin_no++){
        sx_[spin_no].resize(ncells_);
        sy_[spin_no].resize(ncells_);
        sz_[spin_no].resize(ncells_);
    }

    local_density.resize(nbasis_);
    local_density_Mean.resize(nbasis_);
    local_density_square_Mean.resize(nbasis_);
    for (int i = 0; i < local_density.size(); i++)
    {
        local_density[i].resize(2);
        local_density_Mean[i].resize(2);
        local_density_square_Mean[i].resize(2);
    }

    Nematic_order_mean_ = 0.0;
    Nematic_order_square_mean_ = 0.0;

    SiSj_.resize(lx_, ly_);
    SiSj_Mean_.resize(lx_, ly_);
    SiSj_square_Mean_.resize(lx_, ly_);

    SiSjQ_Mean_.resize(lx_, ly_);
    SiSjQ_square_Mean_.resize(lx_, ly_);
    SiSjQ_.resize(lx_, ly_);

    quantum_SiSj_.resize(lx_, ly_);
    quantum_SiSj_Mean_.resize(lx_, ly_);
    quantum_SiSj_square_Mean_.resize(lx_, ly_);

    quantum_SiSjQ_Mean_.resize(lx_, ly_);
    quantum_SiSjQ_square_Mean_.resize(lx_, ly_);
    quantum_SiSjQ_.resize(lx_, ly_);

    for (int ix = 0; ix < lx_; ix++)
    {
        for (int iy = 0; iy < ly_; iy++)
        {
            SiSjQ_Mean_(ix, iy) = zero;
            SiSjQ_square_Mean_(ix, iy) = zero;
        }
    }

} // ----------

double Observables::Omega(int i)
{
    return -20.0 + double(i) * dosincr_;
} // ----------

#endif // OBSERVABLES_H
