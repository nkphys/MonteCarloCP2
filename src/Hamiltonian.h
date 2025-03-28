#include <algorithm>
#include <functional>
#include <math.h>
#include "tensor_type.h"
#include "ParametersEngine.h"
#include "Coordinates.h"
#include "MFParams.h"
#define PI acos(-1.0)

#ifndef Hamiltonian_class
#define Hamiltonian_class

extern "C" void zheev_(char *, char *, int *, std::complex<double> *, int *, double *,
                       std::complex<double> *, int *, double *, int *);

class Hamiltonian
{
public:
    Hamiltonian(Parameters &Parameters__, Coordinates &Coordinates__, Coordinates &CoordinatesCluster__, MFParams &MFParams__)
        : Parameters_(Parameters__), Coordinates_(Coordinates__), CoordinatesCluster_(CoordinatesCluster__), MFParams_(MFParams__)

    {
        Initialize();
      //  Hoppings();
      //  HTBCreate();
      //  HTBClusterCreate();
    }

    void Initialize();                                     //::DONE
    void Hoppings();                                       //::DONE
    double GetCLEnergy();                                  //::DONE
    double Get_Relative_E(int & site_no, int & spin_no, double & theta_old, double & phi_old, double & moment_size_old);
    void InteractionsCreate();                             //::DONE
    void InteractionsClusterCreate(int Center_site);       //::DONE
    void Check_Hermiticity();                              //::DONE
    void Check_up_down_symmetry();                         //::DONE
    void HTBCreate();                                      //::DONE
    void HTBClusterCreate();                               //::DONE
    double chemicalpotential(double muin, double filling); //::DONE

    double chemicalpotentialCluster(double muin, double filling); //::DONE

    double TotalDensity();   //::DONE
    double ClusterDensity(); //::DONE
    double E_QM();           //::DONE

    double E_QMCluster();                 //::DONE
    void Diagonalize(char option);        //::DONE
    void DiagonalizeCluster(char option); //::DONE
    void copy_eigs(int i);                //::DONE
    void copy_eigs_Cluster(int i);        //::DONE

    Parameters &Parameters_;
    Coordinates &Coordinates_;
    Coordinates &CoordinatesCluster_;
    MFParams &MFParams_;
    int lx_, ly_, ncells_, n_orbs_, n_Spins_;
    int lx_cluster_, ly_cluster_, ncells_cluster;
    Matrix<complex<double>> HTB_;
    Matrix<complex<double>> HTBCluster_;
    Matrix<complex<double>> Ham_;
    Matrix<complex<double>> HamCluster_;
    Matrix<double> Tx, Ty, Tpxpy, Tpxmy;
    vector<double> eigs_, eigsCluster_, eigsCluster_saved_, eigs_saved_;
    Mat_2_doub sx_, sy_, sz_;
    Matrix<double> IntraCell_Hopp, InterCell_px_Hopp, InterCell_py_Hopp, InterCell_pxmy_Hopp ;


    double HS_factor;
};

double Hamiltonian::chemicalpotential(double muin, double filling)
{
    double mu_out;
    double n1, N;
    double dMubydN;
    double nstate = eigs_.size();
    dMubydN = 0.005 * (eigs_[nstate - 1] - eigs_[0]) / nstate;
    N = filling * double(eigs_.size());
    //temp=Parameters_.temp;
    mu_out = muin;
    bool converged = false;

    if (!Parameters_.fix_mu)
    {
        assert(!Parameters_.fix_mu);

        if (1 == 2)
        {
            for (int i = 0; i < 100000; i++)
            {
                n1 = 0.0;
                for (int j = 0; j < nstate; j++)
                {
                    n1 += double(1.0 / (exp((eigs_[j] - mu_out) * Parameters_.beta) + 1.0));
                }
                //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
                if (abs(N - n1) < double(0.00001))
                {
                    //cout<<abs(N-n1)<<endl;
                    converged = true;
                    break;
                }
                else
                {
                    mu_out += (N - n1) * dMubydN;
                    //cout<<i<<"    "<<n1<<"    "<<N-n1<<endl;
                }
            }

            if (!converged)
            {
                cout << "mu_not_converged, N = " << n1 << endl;
            }
            else
            {
                //cout<<"mu converged, N = "<<n1<<endl;
            }
        }

        double mu1, mu2;
        double mu_temp = muin;
        //cout<<"mu_input = "<<mu_temp<<endl;
        if (1 == 1)
        {
            mu1 = eigs_[0]-20;
            mu2 = eigs_[nstate - 1]+20;
            for (int i = 0; i < 4000000; i++)
            {
                n1 = 0.0;
                for (int j = 0; j < nstate; j++)
                {
                    n1 += double(1.0 / (exp((eigs_[j] - mu_temp) * Parameters_.beta) + 1.0));
                }
                //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_temp<< endl;
                if (abs(N - n1) < double(0.00001))
                {
                    //cout<<abs(N-n1)<<endl;
                    converged = true;
                    break;
                }
                else
                {
                    if (n1 > N)
                    {
                        mu2 = mu_temp;
                        mu_temp = 0.5 * (mu1 + mu_temp);
                    }
                    else
                    {
                        mu1 = mu_temp;
                        mu_temp = 0.5 * (mu2 + mu_temp);
                    }
                }
                //cout<<"mu_temp = "<<mu_temp<<"   "<<n1<<endl;
            }

            if (!converged)
            {
                cout << "mu_not_converged, N = " << n1 << endl;
            }
            else
            {
                //cout<<"mu converged, N = "<<n1<<endl;
            }

            mu_out = mu_temp;
        }

        return mu_out;
    }
    else
    {
        assert(Parameters_.fix_mu);
        return Parameters_.fixed_mu_value;
    }
} // ----------

double Hamiltonian::chemicalpotentialCluster(double muin, double filling)
{
    double mu_out;
    double n1, N;
    double dMubydN;
    double nstate = eigsCluster_.size();
    dMubydN = 0.005 * (eigsCluster_[nstate - 1] - eigsCluster_[0]) / nstate;
    N = filling * double(eigsCluster_.size());
    //temp=Parameters_.temp;
    mu_out = muin;
    bool converged = false;

    if (!Parameters_.fix_mu)
    {
        assert(!Parameters_.fix_mu);
        if (1 == 2)
        {
            for (int i = 0; i < 100000; i++)
            {
                n1 = 0.0;
                for (int j = 0; j < nstate; j++)
                {
                    n1 += double(1.0 / (exp((eigsCluster_[j] - mu_out) * Parameters_.beta) + 1.0));
                }
                //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
                if (abs(N - n1) < double(0.00001))
                {
                    //cout<<abs(N-n1)<<endl;
                    converged = true;
                    break;
                }
                else
                {
                    mu_out += (N - n1) * dMubydN;
                    //cout<<i<<"    "<<n1<<"    "<<N-n1<<endl;
                }
            }

            if (!converged)
            {
                cout << "mu_not_converged, N = " << n1 << endl;
            }
            else
            {
                //cout<<"mu converged, N = "<<n1<<endl;
            }
        }

        double mu1, mu2;
        double mu_temp = muin;
        //cout<<"mu_input = "<<mu_temp<<endl;
        if (1 == 1)
        {
            mu1 = eigsCluster_[0]-20;
            mu2 = eigsCluster_[nstate - 1]+20;
            for (int i = 0; i < 4000000; i++)
            {
                n1 = 0.0;
                for (int j = 0; j < nstate; j++)
                {
                    n1 += double(1.0 / (exp((eigsCluster_[j] - mu_temp) * Parameters_.beta) + 1.0));
                }
                //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
                if (abs(N - n1) < double(0.00001))
                {
                    //cout<<abs(N-n1)<<endl;
                    converged = true;
                    break;
                }
                else
                {
                    if (n1 > N)
                    {
                        mu2 = mu_temp;
                        mu_temp = 0.5 * (mu1 + mu_temp);
                    }
                    else
                    {
                        mu1 = mu_temp;
                        mu_temp = 0.5 * (mu2 + mu_temp);
                    }
                }
                //cout<<"mu_temp = "<<mu_temp<<"   "<<n1<<endl;
            }

            if (!converged)
            {
                cout << "mu_not_converged, N = " << n1 << endl;
            }
            else
            {
                //cout<<"mu converged, N = "<<n1<<endl;
            }

            mu_out = mu_temp;
        }

        return mu_out;
    }
    else
    {
        assert(Parameters_.fix_mu);
        return Parameters_.fixed_mu_value;
    }
} // ----------

void Hamiltonian::Initialize()
{

    ly_cluster_ = Parameters_.ly_cluster;
    lx_cluster_ = Parameters_.lx_cluster;
    ncells_cluster = lx_cluster_*ly_cluster_;

    ly_ = Parameters_.ly;
    lx_ = Parameters_.lx;
    ncells_ = lx_*ly_;
    n_orbs_ = Parameters_.n_orbs;
    n_Spins_ = Parameters_.n_Spins;
    int space = 2 * ncells_ * n_orbs_;
    int spaceCluster = 2 * ncells_cluster* n_orbs_;



   if(!Parameters_.IgnoreFermions){
    HTB_.resize(space, space);
    Ham_.resize(space, space);
    HTBCluster_.resize(spaceCluster, spaceCluster);
    HamCluster_.resize(spaceCluster, spaceCluster);
    eigs_.resize(space);


    sx_.resize(n_Spins_);
    sy_.resize(n_Spins_);
    sz_.resize(n_Spins_);
    for(int Spin_no=0;Spin_no<n_Spins_;Spin_no++){
        sx_[Spin_no].resize(ncells_);
        sy_[Spin_no].resize(ncells_);
        sz_[Spin_no].resize(ncells_);
    }

    eigs_saved_.resize(space);
    eigsCluster_.resize(spaceCluster);
    eigsCluster_saved_.resize(spaceCluster);


    Hoppings();
    HTBCreate();
    HTBClusterCreate();
    }

} // ----------

double Hamiltonian::TotalDensity()
{
    double n1 = 0.0;
    for (int j = 0; j < eigs_.size(); j++)
    {
        n1 += 1.0 / (exp(Parameters_.beta * (eigs_[j] - Parameters_.mus * 1.0)) + 1.0);
    }
    return n1;
} // ----------

double Hamiltonian::ClusterDensity()
{
    double n1 = 0.0;
    for (int j = 0; j < eigsCluster_.size(); j++)
    {
        n1 += 1.0 / (exp(Parameters_.beta * (eigsCluster_[j] - Parameters_.mus_Cluster * 1.0)) + 1.0);
    }
    return n1;
} // ----------

double Hamiltonian::E_QM()
{
    double E = 0.0;
    for (int j = 0; j < eigs_.size(); j++)
    {
        E += (eigs_[j]) / (exp(Parameters_.beta * (eigs_[j] - Parameters_.mus)) + 1.0);
    }
    return E;
} // ----------

double Hamiltonian::E_QMCluster()
{
    double E = 0.0;
    for (int j = 0; j < eigsCluster_.size(); j++)
    {
        E += (eigsCluster_[j]) / (exp(Parameters_.beta * (eigsCluster_[j] - Parameters_.mus_Cluster)) + 1.0);
    }
    return E;
} // ----------


double Hamiltonian::Get_Relative_E(int & site_no, int & spin_no, double & theta_old, double & phi_old, double & moment_size_old){



double EPS_=0.00000001;
assert(n_Spins_==1);
double EClassical=0.0;
int Spin_i=spin_no;
int i=site_no;
int _ix, _iy;
double ei,ai;
int cell;

Mat_1_doub spins_cell_old, spins_cell, spins_neigh;
spins_cell_old.resize(3);spins_cell.resize(3); spins_neigh.resize(3);

double sx_site, sy_site, sz_site, sx_old_site, sy_old_site, sz_old_site;
 _ix = Coordinates_.indx_cellwise(i);
 _iy = Coordinates_.indy_cellwise(i);

        ei = MFParams_.etheta[Spin_i](_ix,_iy);
        ai = MFParams_.ephi[Spin_i](_ix, _iy);
       
	spins_cell[0]=MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei); 
        spins_cell[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
        spins_cell[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);

	
	spins_cell_old[0] = moment_size_old * cos(phi_old) * sin(theta_old);
        spins_cell_old[1] = moment_size_old * sin(phi_old) * sin(theta_old);
        spins_cell_old[2] = moment_size_old * cos(theta_old);


	
		//magnetic field
		EClassical += -Parameters_.hz_mag*(spins_cell[2]-spins_cell_old[2]);	


                cell = Coordinates_.neigh(i, 0); //+x
		_ix = Coordinates_.indx_cellwise(cell);
                _iy = Coordinates_.indy_cellwise(cell);
                ei = MFParams_.etheta[Spin_i](_ix,_iy);
                ai = MFParams_.ephi[Spin_i](_ix, _iy);
                spins_neigh[0] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
                spins_neigh[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
                spins_neigh[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);
               for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
		if(abs(Parameters_.J_px(c1,c2))>EPS_){
                EClassical += 1.0 *(  Parameters_.J_px(c1,c2)*(spins_cell[c1]-spins_cell_old[c1])*spins_neigh[c2]);
		}
                }
                }
 

		
                cell = Coordinates_.neigh(i, 2); //+y
		_ix = Coordinates_.indx_cellwise(cell);
                _iy = Coordinates_.indy_cellwise(cell);
                ei = MFParams_.etheta[Spin_i](_ix,_iy);
                ai = MFParams_.ephi[Spin_i](_ix, _iy);
                spins_neigh[0] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
                spins_neigh[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
                spins_neigh[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);
               for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
		if(abs(Parameters_.J_py(c1,c2))>EPS_){

                EClassical += 1.0 *(  Parameters_.J_py(c1,c2)*(spins_cell[c1]-spins_cell_old[c1])*spins_neigh[c2]);
		}
                }
                }		





                cell = Coordinates_.neigh(i,5); //mxpy
		_ix = Coordinates_.indx_cellwise(cell);
                _iy = Coordinates_.indy_cellwise(cell);
                ei = MFParams_.etheta[Spin_i](_ix,_iy);
                ai = MFParams_.ephi[Spin_i](_ix, _iy);
                spins_neigh[0] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
                spins_neigh[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
                spins_neigh[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);
               for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
		if(abs(Parameters_.J_mxpy(c1,c2))>EPS_){

                EClassical += 1.0 *(  Parameters_.J_mxpy(c1,c2)*(spins_cell[c1]-spins_cell_old[c1])*spins_neigh[c2]);
               }
		 }
                }





	        cell = Coordinates_.neigh(i, 1); //-x	
		_ix = Coordinates_.indx_cellwise(cell);
                _iy = Coordinates_.indy_cellwise(cell);
                ei = MFParams_.etheta[Spin_i](_ix,_iy);
                ai = MFParams_.ephi[Spin_i](_ix, _iy);
                spins_neigh[0] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
                spins_neigh[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
                spins_neigh[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);
               for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
		if(abs(Parameters_.J_px(c1,c2))>EPS_){

                EClassical += 1.0 *(  Parameters_.J_px(c2,c1)*(spins_cell[c1]-spins_cell_old[c1])*spins_neigh[c2]);
                }
		}
                }		





                cell = Coordinates_.neigh(i, 3); //-y
		_ix = Coordinates_.indx_cellwise(cell);
                _iy = Coordinates_.indy_cellwise(cell);
                ei = MFParams_.etheta[Spin_i](_ix,_iy);
                ai = MFParams_.ephi[Spin_i](_ix, _iy);
                spins_neigh[0] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
                spins_neigh[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
                spins_neigh[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);
               for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
                if(abs(Parameters_.J_py(c1,c2))>EPS_){

		EClassical += 1.0 *(  Parameters_.J_py(c2,c1)*(spins_cell[c1]-spins_cell_old[c1])*spins_neigh[c2]);
    		}
	            }
                }




                cell = Coordinates_.neigh(i,7); //pxmy
		_ix = Coordinates_.indx_cellwise(cell);
                _iy = Coordinates_.indy_cellwise(cell);
                ei = MFParams_.etheta[Spin_i](_ix,_iy);
                ai = MFParams_.ephi[Spin_i](_ix, _iy);
                spins_neigh[0] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
                spins_neigh[1] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
                spins_neigh[2] = MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);
               for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
		if(abs(Parameters_.J_mxpy(c1,c2))>EPS_){

                EClassical += 1.0 *(  Parameters_.J_mxpy(c2,c1)*(spins_cell[c1]-spins_cell_old[c1])*spins_neigh[c2]);
		}
                }
                }



return EClassical;
}


double Hamiltonian::GetCLEnergy()
{


    assert(n_Spins_==1);
    double EClassical;
    int cell;
    double ei, ai, ej, aj;

    Mat_1_doub spins_neigh, spins_cell;
    spins_neigh.resize(3); spins_cell.resize(3);	


    // Classical Energy
    EClassical = double(0.0);

    int _ix, _iy;
    int _jx, _jy;
    for (int i = 0; i < ncells_; i++)
    {
	_ix = Coordinates_.indx_cellwise(i);
        _iy = Coordinates_.indy_cellwise(i);


        for(int Spin_i=0;Spin_i<n_Spins_;Spin_i++){
		ei = MFParams_.etheta[Spin_i](_ix, _iy);
         	ai = MFParams_.ephi[Spin_i](_ix, _iy);
		
		spins_cell[0]=MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ai) * sin(ei);
		spins_cell[1]=MFParams_.Moment_Size[Spin_i](_ix, _iy) * sin(ai) * sin(ei);
		spins_cell[2]=MFParams_.Moment_Size[Spin_i](_ix, _iy) * cos(ei);


                //On-site b/w classical spins,
                
/*
		cell = i;
                EClassical += 1.0 * Parameters_.K_0X_0Y(Spin_i,Spin_j)*( (sx_[Spin_i][i] * sx_[Spin_j][cell]) + (sy_[Spin_i][i] * sy_[Spin_j][cell]) + (1.0 * sz_[Spin_i][i] * sz_[Spin_j][cell]));


                cell = Coordinates_.neigh(i, 0); //+x
                EClassical += 1.0 * Parameters_.K_1X_0Y(Spin_i,Spin_j)*( (sx_[Spin_i][i] * sx_[Spin_j][cell]) + (sy_[Spin_i][i] * sy_[Spin_j][cell]) + (1.0 * sz_[Spin_i][i] * sz_[Spin_j][cell]));


                cell = Coordinates_.neigh(i, 2); //+y
                EClassical += Parameters_.K_0X_1Y(Spin_i,Spin_j) * ((sx_[Spin_i][i] * sx_[Spin_j][cell]) + (sy_[Spin_i][i] * sy_[Spin_j][cell]) + (1.0 * sz_[Spin_i][i] * sz_[Spin_j][cell]));

                cell = Coordinates_.neigh(i,5); //mxpy
                EClassical += Parameters_.K_m1X_1Y(Spin_i,Spin_j) * ((sx_[Spin_i][i] * sx_[Spin_j][cell]) + (sy_[Spin_i][i] * sy_[Spin_j][cell]) + (1.0 * sz_[Spin_i][i] * sz_[Spin_j][cell]));

*/


		EClassical += -Parameters_.hz_mag*spins_cell[2];



		 cell = Coordinates_.neigh(i, 0); //+x
                _jx = Coordinates_.indx_cellwise(cell);
                _jy = Coordinates_.indy_cellwise(cell);
                ej = MFParams_.etheta[Spin_i](_jx, _jy);
                aj = MFParams_.ephi[Spin_i](_jx, _jy);
		spins_neigh[0]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * cos(aj) * sin(ej);
                spins_neigh[1]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * sin(aj) * sin(ej);
                spins_neigh[2]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * cos(ej);
		for(int c1=0;c1<3;c1++){  //x,y,z
		for(int c2=0;c2<3;c2++){
		EClassical += 1.0 *(  Parameters_.J_px(c1,c2)*spins_cell[c1]*spins_neigh[c2]);
		}
		}

		

                cell = Coordinates_.neigh(i, 2); //+y
		_jx = Coordinates_.indx_cellwise(cell);
                _jy = Coordinates_.indy_cellwise(cell);
                ej = MFParams_.etheta[Spin_i](_jx, _jy);
                aj = MFParams_.ephi[Spin_i](_jx, _jy);
                spins_neigh[0]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * cos(aj) * sin(ej);
                spins_neigh[1]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * sin(aj) * sin(ej);
                spins_neigh[2]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * cos(ej);
                for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
                EClassical += 1.0 *(  Parameters_.J_py(c1,c2)*spins_cell[c1]*spins_neigh[c2]);
                }
                }
		




                cell = Coordinates_.neigh(i,5); //mxpy
		_jx = Coordinates_.indx_cellwise(cell);
                _jy = Coordinates_.indy_cellwise(cell);
                ej = MFParams_.etheta[Spin_i](_jx, _jy);
                aj = MFParams_.ephi[Spin_i](_jx, _jy);
                spins_neigh[0]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * cos(aj) * sin(ej);
                spins_neigh[1]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * sin(aj) * sin(ej);
                spins_neigh[2]=MFParams_.Moment_Size[Spin_i](_jx, _jy) * cos(ej);
                for(int c1=0;c1<3;c1++){  //x,y,z
                for(int c2=0;c2<3;c2++){
                EClassical += 1.0 *(  Parameters_.J_mxpy(c1,c2)*spins_cell[c1]*spins_neigh[c2]);
                }
                }






        }
    }



    return EClassical;
} // ----------

void Hamiltonian::InteractionsCreate()
{

    int a;
    double ei, ai;
    int index;
    int i_posx, i_posy;
    int Spin_no;

    Ham_ = HTB_;
    // Ham_.print();

    for (int i = 0; i < ncells_; i++)
    { // For each cell
        i_posx = Coordinates_.indx_cellwise(i);
        i_posy = Coordinates_.indy_cellwise(i);


        for(int orb=0;orb<n_orbs_;orb++){

            index=Coordinates_.Nbasis(i_posx, i_posy, orb);

            if(n_Spins_==n_orbs_){
            Spin_no=orb;
            }
            else{Spin_no=0;}

            ei = MFParams_.etheta[Spin_no](i_posx, i_posy);
            ai = MFParams_.ephi[Spin_no](i_posx, i_posy);


            Ham_(index, index) += Parameters_.J_Hund[orb] * (cos(ei)) * 0.5 * MFParams_.Moment_Size[Spin_no](i_posx, i_posy);
            Ham_(index + (ncells_*n_orbs_), index + (ncells_*n_orbs_)) += Parameters_.J_Hund[orb] * (-cos(ei)) * 0.5 * MFParams_.Moment_Size[Spin_no](i_posx, i_posy);
            Ham_(index, index + (ncells_*n_orbs_)) += Parameters_.J_Hund[orb] * sin(ei) * complex<double>(cos(ai), -sin(ai)) * 0.5 * MFParams_.Moment_Size[Spin_no](i_posx, i_posy); //S-
            Ham_(index + (ncells_*n_orbs_), index) += Parameters_.J_Hund[orb] * sin(ei) * complex<double>(cos(ai), sin(ai)) * 0.5 * MFParams_.Moment_Size[Spin_no](i_posx, i_posy);  //S+

            // On-Site potential
            for (int spin = 0; spin < 2; spin++)
            {
                a = Coordinates_.Nbasis(i_posx,i_posy,orb) + ncells_*n_orbs_*spin;
                Ham_(a, a) += complex<double>(1.0, 0.0) * (
                            Parameters_.OnSiteE[orb] +
                            MFParams_.Disorder(i_posx, i_posy)
                            );
            }
        }
    }

} // ----------

void Hamiltonian::InteractionsClusterCreate(int Center_site)
{

    int x_pos, y_pos;
    double ei, ai;
    int a;
    int i_original;
    int index;
    int i_posx, i_posy, Spin_no;

    HamCluster_ = HTBCluster_;

    for (int i = 0; i < ncells_cluster; i++)
    { // For each cell in cluster

        i_posx = CoordinatesCluster_.indx_cellwise(i);
        i_posy = CoordinatesCluster_.indy_cellwise(i);

        x_pos = Coordinates_.indx_cellwise(Center_site) - int(Parameters_.lx_cluster / 2) + CoordinatesCluster_.indx_cellwise(i);
        y_pos = Coordinates_.indy_cellwise(Center_site) - int(Parameters_.ly_cluster / 2) + CoordinatesCluster_.indy_cellwise(i);
        x_pos = (x_pos + Coordinates_.lx_) % Coordinates_.lx_;
        y_pos = (y_pos + Coordinates_.ly_) % Coordinates_.ly_;

        i_original=Coordinates_.Ncell(x_pos, y_pos);


        for(int orb=0;orb<n_orbs_;orb++){


            if(n_Spins_==n_orbs_){
            Spin_no==orb;
            }
            else{Spin_no=0;}

            ei = MFParams_.etheta[Spin_no](x_pos, y_pos);
            ai = MFParams_.ephi[Spin_no](x_pos, y_pos);

            index=CoordinatesCluster_.Nbasis(i_posx, i_posy, orb);

            HamCluster_(index, index) += Parameters_.J_Hund[orb] * (cos(ei)) * 0.5 * MFParams_.Moment_Size[Spin_no](x_pos, y_pos);
            HamCluster_(index + ncells_cluster*n_orbs_, index + ncells_cluster*n_orbs_) += Parameters_.J_Hund[orb] * (-cos(ei)) * 0.5 * MFParams_.Moment_Size[Spin_no](x_pos, y_pos);
            HamCluster_(index, index + ncells_cluster*n_orbs_) += Parameters_.J_Hund[orb] * sin(ei) * complex<double>(cos(ai), -sin(ai)) * 0.5 * MFParams_.Moment_Size[Spin_no](x_pos, y_pos); //S-
            HamCluster_(index + ncells_cluster*n_orbs_, index) += Parameters_.J_Hund[orb] * sin(ei) * complex<double>(cos(ai), sin(ai)) * 0.5 * MFParams_.Moment_Size[Spin_no](x_pos, y_pos);  //S+


            for (int spin = 0; spin < 2; spin++)
            {
                a = CoordinatesCluster_.Nbasis(i_posx,i_posy,orb) + ncells_cluster*n_orbs_*spin;
                HamCluster_(a, a) += complex<double>(1.0, 0.0) * (
                            Parameters_.OnSiteE[orb] +
                            MFParams_.Disorder(x_pos, y_pos)
                            );
            }
        }
    }


} // ----------

void Hamiltonian::Check_up_down_symmetry()

{
    complex<double> temp(0, 0);
    complex<double> temp2;

    for (int i = 0; i < ncells_*n_orbs_; i++)
    {
        for (int j = 0; j < ncells_*n_orbs_; j++)
        {
            temp2 = Ham_(i, j) - Ham_(i + ncells_*n_orbs_, j + ncells_*n_orbs_); //+ Ham_(i+orbs_*ns_,j) + Ham_(i,j+orbs_*ns_);
            temp += temp2 * conj(temp2);
        }
    }

    cout << "Assymetry in up-down sector: " << temp << endl;
}

void Hamiltonian::Check_Hermiticity()

{
    complex<double> temp(0, 0);
    complex<double> temp2;

    for (int i = 0; i < HamCluster_.n_row(); i++)
    {
        for (int j = 0; j < HamCluster_.n_row(); j++)
        {
            if (HamCluster_(i, j) != conj(HamCluster_(j, i)))
            {
                cout << i << "," << j << endl;
                cout << "i,j = " << HamCluster_(i, j) << ", j,i=" << conj(HamCluster_(j, i)) << endl;
            }
            assert(HamCluster_(i, j) == conj(HamCluster_(j, i))); //+ Ham_(i+orbs_*ns_,j) + Ham_(i,j+orbs_*ns_);
            //temp +=temp2*conj(temp2);
        }
    }

    // cout<<"Hermiticity: "<<temp<<endl;
}

void Hamiltonian::Diagonalize(char option)
{

    //extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
    //                       std::complex<double> *,int *, double *, int *);

    char jobz = option;
    // jobz = 'V';
    char uplo = 'U'; //WHY ONLY 'L' WORKS?
    int n = Ham_.n_row();
    int lda = Ham_.n_col();
    vector<complex<double>> work(3);
    vector<double> rwork(3 * n - 2);
    int info;
    int lwork = -1;

    eigs_.resize(Ham_.n_row());
    fill(eigs_.begin(), eigs_.end(), 0);
    // query:
    zheev_(&jobz, &uplo, &n, &(Ham_(0, 0)), &lda, &(eigs_[0]), &(work[0]), &lwork, &(rwork[0]), &info);
    //lwork = int(real(work[0]))+1;
    lwork = int((work[0].real()));
    work.resize(lwork);
    // real work:
    zheev_(&jobz, &uplo, &n, &(Ham_(0, 0)), &lda, &(eigs_[0]), &(work[0]), &lwork, &(rwork[0]), &info);
    if (info != 0)
    {
        std::cerr << "info=" << info << "\n";
        perror("diag: zheev: failed with info!=0.\n");
    }

    // Ham_.print();

    //  for(int i=0;i<eigs_.size();i++){
    //    cout<<eigs_[i]<<endl;
    //}
}

void Hamiltonian::DiagonalizeCluster(char option)
{

    //extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
    //                       std::complex<double> *,int *, double *, int *);

    char jobz = option;
    // jobz = 'V';
    // cout << option;
    char uplo = 'U'; //WHY ONLY 'L' WORKS?
    int n = HamCluster_.n_row();
    int lda = HamCluster_.n_col();
    vector<complex<double>> work(3);
    vector<double> rwork(3 * n - 2);
    int info;
    int lwork = -1;

    eigsCluster_.resize(HamCluster_.n_row());
    fill(eigsCluster_.begin(), eigsCluster_.end(), 0);
    // query:
    zheev_(&jobz, &uplo, &n, &(HamCluster_(0, 0)), &lda, &(eigsCluster_[0]), &(work[0]), &lwork, &(rwork[0]), &info);
    //lwork = int(real(work[0]))+1;
    lwork = int((work[0].real()));
    work.resize(lwork);
    // real work:
    zheev_(&jobz, &uplo, &n, &(HamCluster_(0, 0)), &lda, &(eigsCluster_[0]), &(work[0]), &lwork, &(rwork[0]), &info);
    if (info != 0)
    {
        std::cerr << "info=" << info << "\n";
        perror("diag: zheev: failed with info!=0.\n");
    }

    // Ham_.print();

    //  for(int i=0;i<eigs_.size();i++){
    //    cout<<eigs_[i]<<endl;
    //}
}

void Hamiltonian::HTBCreate()
{

    //Convention used
    //orb=0=d
    //orb=1=px
    //orb=2=py

    int mx = Parameters_.TBC_mx;
    int my = Parameters_.TBC_my;

    complex<double> phasex, phasey;
    int l, m, a, b;
    int lx_pos, ly_pos;
    int mx_pos, my_pos;

    complex<double> TBC_phasex_TS, TBC_phasey_TS;

    HTB_.fill(0.0);

    for (l = 0; l < ncells_; l++)
    {
        lx_pos = Coordinates_.indx_cellwise(l);
        ly_pos = Coordinates_.indy_cellwise(l);



        //On-site Hopping
        m = l; //same cell
        mx_pos = Coordinates_.indx_cellwise(m);
        my_pos = Coordinates_.indy_cellwise(m);

        for (int spin=0; spin<2; spin++){
            for(int orb1=0;orb1<n_orbs_;orb1++){
                for(int orb2=0;orb2<n_orbs_;orb2++){
                    if(Parameters_.hopping_0X_0Y(orb1,orb2)!=0.0){
                        a = Coordinates_.Nbasis(lx_pos,ly_pos,orb1) + ncells_*n_orbs_*spin;
                        b = Coordinates_.Nbasis(mx_pos,my_pos,orb2) + ncells_*n_orbs_*spin;
                        assert(a != b);
                        if (a != b)
                        {
                            HTB_(b, a) = complex<double>(1.0 *Parameters_.hopping_0X_0Y(orb1,orb2), 0.0);
                            HTB_(a, b) = conj(HTB_(b, a));
                        }
                    }
                }
            }
        }


        // * +x direction Neighbor
        if (lx_pos == (Coordinates_.lx_ - 1))
        {
            phasex = Parameters_.BoundaryConnection*one_complex;//exp(iota_complex * 2.0 * (1.0 * mx) * PI / (1.0 * Parameters_.TBC_cellsX));
            phasey = one_complex;
        }
        else
        {
            phasex = one_complex;
            phasey = one_complex;
        }


        TBC_phasex_TS = exp((1.0/Coordinates_.lx_ )*(iota_complex * 2.0 * (1.0 * mx) * PI / (1.0 * Parameters_.TBC_cellsX)));
        m = Coordinates_.neigh(l, 0); //+x neighbour cell
        mx_pos = Coordinates_.indx_cellwise(m);
        my_pos = Coordinates_.indy_cellwise(m);

        for (int spin=0; spin<2; spin++){
            for(int orb1=0;orb1<n_orbs_;orb1++){
                for(int orb2=0;orb2<n_orbs_;orb2++){
                    if(Parameters_.hopping_1X_0Y(orb1,orb2)!=0.0){
                        a = Coordinates_.Nbasis(lx_pos,ly_pos,orb1) + ncells_*n_orbs_*spin;
                        b = Coordinates_.Nbasis(mx_pos,my_pos,orb2) + ncells_*n_orbs_*spin;
                        assert(a != b);
                        if (a != b)
                        {
                            HTB_(b, a) = complex<double>(1.0 *Parameters_.hopping_1X_0Y(orb1,orb2), 0.0) * phasex*TBC_phasex_TS;
                            HTB_(a, b) = conj(HTB_(b, a));
                        }
                    }
                }
            }
        }


        if(Coordinates_.ly_>1){


            // * +y direction Neighbor
            if (ly_pos == (Coordinates_.ly_ - 1))
            {
                phasex = one_complex;
                phasey = Parameters_.BoundaryConnection*one_complex;//exp(iota_complex * 2.0 * (1.0 * my) * PI / (1.0 * Parameters_.TBC_cellsY));
            }
            else
            {
                phasex = one_complex;
                phasey = one_complex;
            }

            TBC_phasey_TS = exp((1.0/Coordinates_.ly_ )*(iota_complex * 2.0 * (1.0 * my) * PI / (1.0 * Parameters_.TBC_cellsY)));

            m = Coordinates_.neigh(l, 2); //+y neighbour cell
            mx_pos = Coordinates_.indx_cellwise(m);
            my_pos = Coordinates_.indy_cellwise(m);

            for (int spin = 0; spin < 2; spin++){
                for(int orb1=0;orb1<n_orbs_;orb1++){
                    for(int orb2=0;orb2<n_orbs_;orb2++){
                        if(Parameters_.hopping_0X_1Y(orb1,orb2)!=0.0){

                            a = Coordinates_.Nbasis(lx_pos,ly_pos,orb1) + ncells_*n_orbs_*spin;
                            b = Coordinates_.Nbasis(mx_pos,my_pos,orb2) + ncells_*n_orbs_*spin;
                            assert(a != b);
                            if (a != b)
                            {
                                HTB_(b, a) = complex<double>(1.0*Parameters_.hopping_0X_1Y(orb1,orb2), 0.0) * phasey*TBC_phasey_TS;
                                HTB_(a, b) = conj(HTB_(b, a));
                            }
                        }
                    }
                }
            }




            // * -x+y direction Neighbor
            if (ly_pos == (Coordinates_.ly_ - 1))
            {
                phasey = Parameters_.BoundaryConnection*one_complex;//exp(iota_complex * 2.0 * (1.0 * my) * PI / (1.0 * Parameters_.TBC_cellsY));
            }
            else
            {
                phasey = one_complex;
            }

            if(lx_pos==0){
                phasex = Parameters_.BoundaryConnection*one_complex;//exp(iota_complex * 2.0 * (-1.0 * mx) * PI / (1.0 * Parameters_.TBC_cellsX));
            }
            else{
                phasex=one_complex;
            }

            TBC_phasex_TS = exp((1.0/Coordinates_.lx_ )*(iota_complex * 2.0 * (-1.0 * mx) * PI / (1.0 * Parameters_.TBC_cellsX)));
            TBC_phasey_TS = exp((1.0/Coordinates_.ly_ )*(iota_complex * 2.0 * (1.0 * my) * PI / (1.0 * Parameters_.TBC_cellsY)));


            m = Coordinates_.neigh(l, 5); //-x+y neighbour cell
            mx_pos = Coordinates_.indx_cellwise(m);
            my_pos = Coordinates_.indy_cellwise(m);

            for (int spin = 0; spin < 2; spin++){
                for(int orb1=0;orb1<n_orbs_;orb1++){
                    for(int orb2=0;orb2<n_orbs_;orb2++){
                        if(Parameters_.hopping_m1X_1Y(orb1,orb2)!=0.0){

                            a = Coordinates_.Nbasis(lx_pos,ly_pos,orb1) + ncells_*n_orbs_*spin;
                            b = Coordinates_.Nbasis(mx_pos,my_pos,orb2) + ncells_*n_orbs_*spin;
                            assert(a != b);
                            if (a != b)
                            {
                                HTB_(b, a) = complex<double>(1.0*Parameters_.hopping_m1X_1Y(orb1,orb2), 0.0) * phasey*phasex*TBC_phasex_TS*TBC_phasey_TS;
                                HTB_(a, b) = conj(HTB_(b, a));
                            }
                        }
                    }
                }
            }



        }



    }




} // ----------

void Hamiltonian::HTBClusterCreate()
{

    if(Parameters_.ED_==false){

        int l, m, a, b;
        int lx_pos, ly_pos;
        int mx_pos, my_pos;

        HTBCluster_.fill(0.0);

        for (l = 0; l < ncells_cluster; l++)
        {

            lx_pos = CoordinatesCluster_.indx_cellwise(l);
            ly_pos = CoordinatesCluster_.indy_cellwise(l);



            // Onsite hopping
            m = l;
            mx_pos = CoordinatesCluster_.indx_cellwise(m);
            my_pos = CoordinatesCluster_.indy_cellwise(m);

            for (int spin = 0; spin < 2; spin++){
                for(int orb1=0;orb1<n_orbs_;orb1++){
                    for(int orb2=0;orb2<n_orbs_;orb2++){
                        if(Parameters_.hopping_0X_0Y(orb1,orb2)!=0.0){
                            a = CoordinatesCluster_.Nbasis(lx_pos,ly_pos,orb1) + ncells_cluster*n_orbs_*spin;
                            b = CoordinatesCluster_.Nbasis(mx_pos,my_pos,orb2) + ncells_cluster*n_orbs_*spin;

                            assert(a != b);
                            if (a != b)
                            {
                                HTBCluster_(b, a) = complex<double>(1.0 *Parameters_.hopping_0X_0Y(orb1,orb2), 0.0);
                                HTBCluster_(a, b) = conj(HTBCluster_(b, a));
                            }
                        }
                    }
                }
            }


            // * +x direction Neighbor
            m = CoordinatesCluster_.neigh(l, 0);
            mx_pos = CoordinatesCluster_.indx_cellwise(m);
            my_pos = CoordinatesCluster_.indy_cellwise(m);

            for (int spin = 0; spin < 2; spin++){
                for(int orb1=0;orb1<n_orbs_;orb1++){
                    for(int orb2=0;orb2<n_orbs_;orb2++){
                        if(Parameters_.hopping_1X_0Y(orb1,orb2)!=0.0){
                            a = CoordinatesCluster_.Nbasis(lx_pos,ly_pos,orb1) + ncells_cluster*n_orbs_*spin;
                            b = CoordinatesCluster_.Nbasis(mx_pos,my_pos,orb2) + ncells_cluster*n_orbs_*spin;

                            assert(a != b);
                            if (a != b)
                            {
                                HTBCluster_(b, a) = complex<double>(1.0 *Parameters_.hopping_1X_0Y(orb1,orb2), 0.0);
                                HTBCluster_(a, b) = conj(HTBCluster_(b, a));
                            }
                        }
                    }
                }
            }


            if(CoordinatesCluster_.ly_>1){

                // * +y direction Neighbor
                m = CoordinatesCluster_.neigh(l, 2);
                mx_pos = CoordinatesCluster_.indx_cellwise(m);
                my_pos = CoordinatesCluster_.indy_cellwise(m);

                for (int spin = 0; spin < 2; spin++){
                    for(int orb1=0;orb1<n_orbs_;orb1++){
                        for(int orb2=0;orb2<n_orbs_;orb2++){
                            if(Parameters_.hopping_0X_1Y(orb1,orb2)!=0.0){
                                a = CoordinatesCluster_.Nbasis(lx_pos,ly_pos,orb1) + ncells_cluster*n_orbs_*spin;
                                b = CoordinatesCluster_.Nbasis(mx_pos,my_pos,orb2) + ncells_cluster*n_orbs_*spin;
                                assert(a != b);
                                if (a != b)
                                {
                                    HTBCluster_(b, a) = complex<double>(1.0 *Parameters_.hopping_0X_1Y(orb1,orb2), 0.0);
                                    HTBCluster_(a, b) = conj(HTBCluster_(b, a));
                                }
                            }
                        }
                    }
                }


                //-x+y direction
                m = CoordinatesCluster_.neigh(l, 5);
                mx_pos = CoordinatesCluster_.indx_cellwise(m);
                my_pos = CoordinatesCluster_.indy_cellwise(m);

                for (int spin = 0; spin < 2; spin++){
                    for(int orb1=0;orb1<n_orbs_;orb1++){
                        for(int orb2=0;orb2<n_orbs_;orb2++){
                            if(Parameters_.hopping_m1X_1Y(orb1,orb2)!=0.0){
                                a = CoordinatesCluster_.Nbasis(lx_pos,ly_pos,orb1) + ncells_cluster*n_orbs_*spin;
                                b = CoordinatesCluster_.Nbasis(mx_pos,my_pos,orb2) + ncells_cluster*n_orbs_*spin;
                                assert(a != b);
                                if (a != b)
                                {
                                    HTBCluster_(b, a) = complex<double>(1.0 *Parameters_.hopping_m1X_1Y(orb1,orb2), 0.0);
                                    HTBCluster_(a, b) = conj(HTBCluster_(b, a));
                                }
                            }
                        }
                    }
                }


            }

        }

    }
    else{
        HTBCluster_=HTB_;
    }

    // HTBCluster_.print();

} // ----------

void Hamiltonian::Hoppings()
{
    //Using matrices from Parameters_

} // ----------

void Hamiltonian::copy_eigs(int i)
{

    int space = 2 * ncells_ *n_orbs_;

    if (i == 0)
    {
        for (int j = 0; j < space; j++)
        {
            eigs_[j] = eigs_saved_[j];
        }
    }
    else
    {
        for (int j = 0; j < space; j++)
        {
            eigs_saved_[j] = eigs_[j];
        }
    }
}

void Hamiltonian::copy_eigs_Cluster(int i)
{

    int space = 2 * ncells_cluster *n_orbs_;

    if (i == 0)
    {
        for (int j = 0; j < space; j++)
        {
            eigsCluster_[j] = eigsCluster_saved_[j];
        }
    }
    else
    {
        for (int j = 0; j < space; j++)
        {
            eigsCluster_saved_[j] = eigsCluster_[j];
        }
    }
}

#endif
