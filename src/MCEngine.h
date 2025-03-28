#include <math.h>
#include "ParametersEngine.h"
#include "tensor_type.h"
#include "random"
#include <stdlib.h>
#define PI acos(-1.0)

#ifndef MCENGINE_H
#define MCENGINE_H

class MCEngine
{
public:
    MCEngine(Parameters &Parameters__, mt19937_64 &Generator1__)
        : Parameters_(Parameters__), Generator1_(Generator1__),
        ns_(Parameters_.ns)
    {
    }


    void Initialize_MarsagliaParams();
    void Update_Dvecs();
    double random1();
    void Create_Connections_wrt_site();
    void FieldThrow(int site);
    void RUN_MC();
    complex<double> GetLocalOprExp(string opr_str, int opr_site);
    double Get_Local_Energy(int site);

    void Calculate_SiSj();
    void Calculate_TauZiTauZj();
    void TauZiTauZj_Average();
    void SiSj_Average();
    void Calculate_TotalE();
    void TotalE_Average();


    Mat_2_int LocalConnectionSize;
    Mat_3_int LocalConnectionSites;
    Mat_3_string LocalConnectionOprs;
    Mat_2_doub LocalConnectionValue;

    int theta1_ind, theta2_ind, phi1_ind, phi2_ind, phi3_ind;
    int X_, Y_, Z_;
    Mat_2_Complex_doub Dvecs;
    Mat_1_doub theta1_, theta2_, phi1_, phi2_, phi3_;
    Parameters &Parameters_;
    mt19937_64 &Generator1_; //for random fields
    const int ns_;


    double TotalE_, TotalE_Mean_, TotalE_square_Mean_;
    Matrix<double> SiSj_, SiSj_Mean_, SiSj_square_Mean_;
    Matrix<double> TauZiTauZj_, TauZiTauZj_Mean_, TauZiTauZj_square_Mean_;

    uniform_real_distribution<double> dis1_; //for random fields

};

/*
 * ***********
 *  Functions in Class MCEngine ------
 *  ***********
*/


void MCEngine::Calculate_TotalE(){

    TotalE_=0.0;

    string temp_opr_str;
    Mat_1_string oprs_list;
    Mat_1_int oprs_site;

    for(int FileNo=0;FileNo<Parameters_.ConnectionFiles.size();FileNo++){
        for(int connection_no=0;connection_no<Parameters_.Connections[FileNo].size();connection_no++){

            stringstream connection_stream;
            connection_stream<<Parameters_.Connections[FileNo][connection_no];

            int n_oprs;
            oprs_list.clear();
            oprs_site.clear();

            double connection_val;
            connection_stream>>n_oprs;
            oprs_list.resize(n_oprs);
            oprs_site.resize(n_oprs);

            for(int opr_no=(n_oprs-1);opr_no>=0;opr_no--){
                connection_stream>>temp_opr_str;
                oprs_list[opr_no]=temp_opr_str;
            }
            for(int opr_no=(n_oprs-1);opr_no>=0;opr_no--){
                connection_stream>>oprs_site[opr_no];
            }
            connection_stream>>connection_val;


            complex<double> E_conn=1.0;
            for(int opr_no=0;opr_no<oprs_list.size();opr_no++){
                int opr_site = oprs_site[opr_no];
                string opr_str = oprs_list[opr_no];
                E_conn = E_conn*GetLocalOprExp(opr_str, opr_site);
            }
            E_conn = E_conn*connection_val;

            assert(E_conn.imag()<0.000001);

            TotalE_ += E_conn.real();

        }}


}

void MCEngine::TotalE_Average(){

    TotalE_Mean_ += TotalE_;
    TotalE_square_Mean_ += TotalE_*TotalE_;

}


void MCEngine::Calculate_SiSj(){

    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            SiSj_(site_i, site_j) =  GetLocalOprExp("Sx", site_i).real()*GetLocalOprExp("Sx", site_j).real()
                                    + GetLocalOprExp("Sy", site_i).real()*GetLocalOprExp("Sy", site_j).real()
                                    + GetLocalOprExp("Sz", site_i).real()*GetLocalOprExp("Sz", site_j).real();
        }
    }

}


void MCEngine::Calculate_TauZiTauZj(){

    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            TauZiTauZj_(site_i, site_j) =  0.25*( GetLocalOprExp("Sx2", site_i).real() - GetLocalOprExp("Sy2", site_i).real() )*
                                    ( GetLocalOprExp("Sx2", site_j).real() - GetLocalOprExp("Sy2", site_j).real() );
        }
    }

}


void MCEngine::TauZiTauZj_Average()
{

    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            TauZiTauZj_Mean_(site_i, site_j) += TauZiTauZj_(site_i, site_j);
            TauZiTauZj_square_Mean_(site_i, site_j) += (TauZiTauZj_(site_i, site_j) * TauZiTauZj_(site_i, site_j));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------



void MCEngine::SiSj_Average()
{

    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            SiSj_Mean_(site_i, site_j) += SiSj_(site_i, site_j);
            SiSj_square_Mean_(site_i, site_j) += (SiSj_(site_i, site_j) * SiSj_(site_i, site_j));
            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------


void MCEngine::Create_Connections_wrt_site(){

    LocalConnectionSize.resize(ns_);
    LocalConnectionSites.resize(ns_);
    LocalConnectionOprs.resize(ns_);
    LocalConnectionValue.resize(ns_);


    string temp_opr_str;
    Mat_1_string oprs_list;
    Mat_1_int oprs_site;

    for(int FileNo=0;FileNo<Parameters_.ConnectionFiles.size();FileNo++){
        for(int connection_no=0;connection_no<Parameters_.Connections[FileNo].size();connection_no++){
            // cout<<"here 2"<<endl;
            // connection_stream.str("");

            stringstream connection_stream;
            connection_stream<<Parameters_.Connections[FileNo][connection_no];

            int n_oprs;
            oprs_list.clear();
            oprs_site.clear();


            double connection_val;
            connection_stream>>n_oprs;
            oprs_list.resize(n_oprs);
            oprs_site.resize(n_oprs);

            for(int opr_no=(n_oprs-1);opr_no>=0;opr_no--){
                connection_stream>>temp_opr_str;
                oprs_list[opr_no]=temp_opr_str;
            }
            for(int opr_no=(n_oprs-1);opr_no>=0;opr_no--){
                connection_stream>>oprs_site[opr_no];
            }
            connection_stream>>connection_val;


            for(int site=0;site<ns_;site++){

                bool site_present=false;
                for(int site_ind=0;site_ind<oprs_site.size();site_ind++){
                    if(site==oprs_site[site_ind]){
                        site_present=true;
                    }
                }

                if(site_present){
                    LocalConnectionSize[site].push_back(oprs_site.size());
                    LocalConnectionValue[site].push_back(connection_val);
                    LocalConnectionSites[site].push_back(oprs_site);
                    LocalConnectionOprs[site].push_back(oprs_list);
                }

            }



        }}



}



double MCEngine::random1()
{
    return dis1_(Generator1_);
}

void MCEngine::Initialize_MarsagliaParams(){

    X_=0;Y_=1;Z_=2;
    theta1_ind=0;theta2_ind=1;
    phi1_ind=2;phi2_ind=3;phi3_ind=4;



    Dvecs.resize(ns_);
    for(int i=0;i<ns_;i++){
        Dvecs[i].resize(3);
    }

    theta1_.resize(ns_);theta2_.resize(ns_);
    phi1_.resize(ns_);phi2_.resize(ns_);phi3_.resize(ns_);


    for(int i=0;i<ns_;i++){
        theta1_[i] = random1();
        theta2_[i] = random1();
        phi1_[i] = 2.0*PI*random1();
        phi2_[i] = 2.0*PI*random1();
        phi3_[i] = 2.0*PI*random1();
    }


    Update_Dvecs();


}

void MCEngine::Update_Dvecs(){


    double x1, x2, y1, y2, z1, z2;

    for(int i=0;i<ns_;i++){
        x1 = sqrt(sqrt(theta2_[i])) * sqrt(theta1_[i]) * sin(phi1_[i]);
        x2 = sqrt(sqrt(theta2_[i])) * sqrt(theta1_[i]) * cos(phi1_[i]);

        y1 = sqrt(sqrt(theta2_[i])) * sqrt(1.0 - theta1_[i]) * sin(phi2_[i]);
        y2 = sqrt(sqrt(theta2_[i])) * sqrt(1.0 - theta1_[i]) * cos(phi2_[i]);

        z1 = sqrt(1.0 - sqrt(theta2_[i])) * sin(phi3_[i]);
        z2 = sqrt(1.0 - sqrt(theta2_[i])) * cos(phi3_[i]);

        Dvecs[i][X_] = complex<double>(x1,x2);
        Dvecs[i][Y_] = complex<double>(y1,y2);
        Dvecs[i][Z_] = complex<double>(z1,z2);
    }

}


void MCEngine::FieldThrow(int site)
{

    theta1_[site] = random1();
    theta2_[site] = random1();
    phi1_[site] = 2.0*PI*random1();
    phi2_[site] = 2.0*PI*random1();
    phi3_[site] = 2.0*PI*random1();

    //	Push_to_Prob_Distributions( etheta[Spin_no](a,b), ephi[Spin_no](a,b) );

} // ----------


complex<double> MCEngine::GetLocalOprExp(string opr_str, int opr_site){

    complex<double> value;

    double x1, x2, y1, y2, z1, z2;
    complex<double> x_,y_,z_;
    x1 = sqrt(sqrt(theta2_[opr_site])) * sqrt(theta1_[opr_site]) * sin(phi1_[opr_site]);
    x2 = sqrt(sqrt(theta2_[opr_site])) * sqrt(theta1_[opr_site]) * cos(phi1_[opr_site]);

    y1 = sqrt(sqrt(theta2_[opr_site])) * sqrt(1.0 - theta1_[opr_site]) * sin(phi2_[opr_site]);
    y2 = sqrt(sqrt(theta2_[opr_site])) * sqrt(1.0 - theta1_[opr_site]) * cos(phi2_[opr_site]);

    z1 = sqrt(1.0 - sqrt(theta2_[opr_site])) * sin(phi3_[opr_site]);
    z2 = sqrt(1.0 - sqrt(theta2_[opr_site])) * cos(phi3_[opr_site]);


    x_=complex<double>(x1,x2);
    y_=complex<double>(y1,y2);
    z_=complex<double>(z1,z2);

    if(opr_str == "Sz"){
        value = iota_complex*( x_*conj(y_) - y_*conj(x_) );
    }
    if(opr_str == "Sx"){
        value = iota_complex*( y_*conj(z_) - z_*conj(y_) );
    }
    if(opr_str == "Sy"){
        value = iota_complex*( z_*conj(x_) - x_*conj(z_) );
    }
    if(opr_str == "iSy"){
        value = -1.0*( z_*conj(x_) - x_*conj(z_) );
    }
    if(opr_str == "Sz2"){
        value = 1.0 - abs(z_)*abs(z_);
    }
    if(opr_str == "Sx2"){
        value = 1.0 - abs(x_)*abs(x_);
    }
    if(opr_str == "Sy2"){
        value = 1.0 - abs(y_)*abs(y_);
    }
    if(opr_str == "Qyz"){
        value = -1.0*(conj(y_)*z_ + conj(z_)*y_);
    }
    if(opr_str == "iQyz"){
        value = -1.0*iota_complex*(conj(y_)*z_ + conj(z_)*y_);
    }
    if(opr_str == "Qxz"){
        value = -1.0*(conj(x_)*z_ + conj(z_)*x_);
    }


    return value;

}

double MCEngine::Get_Local_Energy(int site){

    complex<double> E_local=0;
    for(int conn_no=0;conn_no<LocalConnectionSize[site].size();conn_no++){

        complex<double> E_conn=1.0;
        for(int opr_no=0;opr_no<LocalConnectionSize[site][conn_no];opr_no++){
            int opr_site = LocalConnectionSites[site][conn_no][opr_no];
            string opr_str = LocalConnectionOprs[site][conn_no][opr_no];
            E_conn = E_conn*GetLocalOprExp(opr_str, opr_site);
        }
        E_conn = E_conn*LocalConnectionValue[site][conn_no];

        E_local += E_conn;

    }

    assert(E_local.imag()<0.0000001);

    return E_local.real();
}


void MCEngine::RUN_MC()
{


    SiSj_.resize(ns_,ns_);SiSj_Mean_.resize(ns_,ns_);SiSj_square_Mean_.resize(ns_,ns_);
    TauZiTauZj_.resize(ns_,ns_);TauZiTauZj_Mean_.resize(ns_,ns_);TauZiTauZj_square_Mean_.resize(ns_,ns_);




    int MC_sweeps_used_for_Avg = Parameters_.Last_n_sweeps_for_measurement;
    int Gap_bw_sweeps = Parameters_.Measurement_after_each_m_sweeps;

    double LocalPrevE, LocalCurrE, P12;
    double Prob_check;

    double saved_Params[5];

    double temp_ = Parameters_.temp_max;

    //	assert(false);

    bool guess_accepted;

    for(int temp_point=0;temp_point<Parameters_.Temp_values.size();temp_point++){
        temp_ = Parameters_.Temp_values[temp_point];
        cout << "Temperature = " << temp_ << " is being done" << endl;
        Parameters_.temp = temp_;
        Parameters_.beta = double( 1.0/(Parameters_.Boltzman_constant*temp_));


        char temp_char[50];
        sprintf(temp_char, "%.10f", temp_);

        Parameters_.WindowSize = 0.1; //2f + 0.003f*beta0;
        Parameters_.Eav = 0.0;
        Parameters_.MCNorm = 0;
        Parameters_.Dflag = 'N'; //N // flag to calculate only Eigenvalue
        //std::string name="Output/Conf_" + to_string(ltemp) + ".dat";
        //Parameters_.beta = double(11604.0/ (Parameters_.temp +20.0) );
        //cout << "TEMP  " << Parameters_.temp << endl;



        string File_Out_real_space_corr = "Real_space_corr" + string(temp_char) + ".txt";
        ofstream File_Out_Real_Space_Corr(File_Out_real_space_corr.c_str());


        int Confs_used = 0;
        int measure_start = 0;



        for (int site_i = 0; site_i < ns_; site_i++)
        {
            for (int site_j = 0; site_j <ns_; site_j++)
            {
                SiSj_Mean_(site_i, site_j)=0.0;
                SiSj_square_Mean_(site_i, site_j)=0.0;
                TauZiTauZj_Mean_(site_i, site_j)=0.0;
                TauZiTauZj_square_Mean_(site_i, site_j)=0.0;
        }}

        TotalE_Mean_=0.0;
        TotalE_square_Mean_=0.0;

        for (int count = 0; count < Parameters_.IterMax; count++)
        {

            for (int i = 0; i < ns_; i++)
            { //For each site


                LocalPrevE = Get_Local_Energy(i);

                saved_Params[theta1_ind] = theta1_[i];
                saved_Params[theta2_ind] = theta2_[i];
                saved_Params[phi1_ind] = phi1_[i];
                saved_Params[phi2_ind] = phi2_[i];
                saved_Params[phi3_ind] = phi3_[i];

                FieldThrow(i);
                LocalCurrE = Get_Local_Energy(i);

                P12 = Parameters_.beta * ((LocalPrevE) - (LocalCurrE));

                //Heat bath algorithm [See page-129 of Prof. Elbio's Book]
                //Heat bath algorithm works for small changes i.e. when P~1.0
                //  if (Heat_Bath_Algo){
                //     P =P/(1.0+P);
                //  }

                //Metropolis Algotithm
                // if (Metropolis_Algo){
                //    P=min(1.0,P);
                // }

                if(Parameters_.Metropolis_Algorithm){
                    Prob_check = min(1.0,exp(P12));
                }
                else{ //Heat bath
                    Prob_check =exp(P12)/(1.0+exp(P12));
                }

                /*
       * VON NEUMANN's REJECTING METHOD:
       * Random number < P -----> ACCEPT
       * Random number > P -----> REJECT
       */

                //ACCEPTED
                if (Prob_check > ( random1()) )
                {
                    //PrevE = CurrE;
                }
                //REJECTED
                else
                {
                    theta1_[i] = saved_Params[theta1_ind];
                    theta2_[i] = saved_Params[theta2_ind];
                    phi1_[i] = saved_Params[phi1_ind];
                    phi2_[i] = saved_Params[phi2_ind];
                    phi3_[i] = saved_Params[phi3_ind];
                }

            } // site loop



            //in measurement loops
            if(count > (Parameters_.IterMax - (Gap_bw_sweeps * (MC_sweeps_used_for_Avg - 1) + MC_sweeps_used_for_Avg))){

                if (measure_start == 0)
                {
                    measure_start++;
                    cout << "----------Measurement is started----------" << endl;
                }
                int temp_count = count -
                                 (Parameters_.IterMax - (Gap_bw_sweeps * (MC_sweeps_used_for_Avg - 1) + MC_sweeps_used_for_Avg));
                int zero_or_not = temp_count % (Gap_bw_sweeps + 1);
                if (zero_or_not == 0)
                {
                    Calculate_SiSj();
                    Calculate_TauZiTauZj();
                    SiSj_Average();
                    TauZiTauZj_Average();

                    Calculate_TotalE();
                    TotalE_Average();
                    Confs_used = Confs_used + 1;
                }


            }


        } // Iter Loop


        cout<<"Total "<<Confs_used<<" configurations used for measurement"<<endl;

        File_Out_Real_Space_Corr<<"#i"<<setw(15)<<"j"<<setw(15)<<"<SS(i,j)>"<<setw(15)<<"sd(SS(i,j))"<<setw(15)<<"<TzTz(i,j)>"<<setw(15)<<"sd(TzTz(i,j))"<<endl;
        for (int site_i = 0; site_i < ns_; site_i++)
        {
            for (int site_j = 0; site_j < ns_; site_j++)
            {
                double std_SS = max(0.0,(((SiSj_square_Mean_(site_i, site_j) / (Confs_used * 1.0)) - ((SiSj_Mean_(site_i, site_j) * SiSj_Mean_(site_i, site_j)) / (Confs_used * Confs_used * 1.0)))));
                double std_TzTz = max(0.0,(((TauZiTauZj_square_Mean_(site_i, site_j) / (Confs_used * 1.0)) - ((TauZiTauZj_Mean_(site_i, site_j) * TauZiTauZj_Mean_(site_i, site_j)) / (Confs_used * Confs_used * 1.0)))));
                File_Out_Real_Space_Corr << site_i << setw(15) << site_j << setw(15) << SiSj_Mean_(site_i, site_j) / (Confs_used * 1.0)
                                         << setw(15) << sqrt(std_SS)
                << setw(15) << TauZiTauZj_Mean_(site_i, site_j) / (Confs_used * 1.0)
                                         << setw(15) << sqrt(std_TzTz)
                <<endl;
            }
            File_Out_Real_Space_Corr << endl;
        }



        cout<<" Temperature =  "<<temp_<<" done"<<endl;

        double std_E = max(0.0,(((TotalE_square_Mean_ / (Confs_used * 1.0)) - ((TotalE_Mean_ * TotalE_Mean_) / (Confs_used * Confs_used * 1.0)))));
        cout<<"Temperature, Energy, std(Energy) : "<<setw(15)<<temp_<<setw(15)<<TotalE_Mean_<<setw(15)<<std_E<<endl;


        // string ObsOutFile_str="LocalObsOut_Temperature"+ string(temp_char)+".txt";
        // ofstream ObsOutFile(ObsOutFile_str.c_str());
        // ObsOutFile<<"#site Sx Sy Sz Sx2 Sy2 Sz2"<<endl;
        // for(int site=0;site<ns_;site++){
        //     ObsOutFile<<site<<"  "<<GetLocalOprExp("Sx", site).real()<<"  "<<GetLocalOprExp("Sx", site).imag()<<"  "
        //                     <<GetLocalOprExp("Sy", site).real()<<"  "<<GetLocalOprExp("Sy", site).imag()<<"  "
        //                     <<GetLocalOprExp("Sz", site).real()<<"  "<<GetLocalOprExp("Sz", site).imag()<<"  "
        //                <<GetLocalOprExp("Sx2", site).real()<<"  "<<GetLocalOprExp("Sx2", site).imag()<<"  "
        //                <<GetLocalOprExp("Sy2", site).real()<<"  "<<GetLocalOprExp("Sy2", site).imag()<<"  "
        //                <<GetLocalOprExp("Sz2", site).real()<<"  "<<GetLocalOprExp("Sz2", site).imag()<<"  "
        //                <<endl;
        // }

        // string ObsOutFile2_str="CorrOut_Temperature"+ string(temp_char)+".txt";
        // ofstream ObsOutFile2(ObsOutFile2_str.c_str());
        // ObsOutFile2<<"#site   S.S(0,site)   TauZTauZ(N/2,site)"<<endl;
        // for(int site=0;site<ns_;site++){
        //     double SS =  GetLocalOprExp("Sx", 0).real()*GetLocalOprExp("Sx", site).real()
        //                + GetLocalOprExp("Sy", 0).real()*GetLocalOprExp("Sy", site).real()
        //                 + GetLocalOprExp("Sz", 0).real()*GetLocalOprExp("Sz", site).real();
        //     double TzTz = 0.25*( GetLocalOprExp("Sx2", ns_/2).real() - GetLocalOprExp("Sy2", ns_/2).real() )*
        //                   ( GetLocalOprExp("Sx2", site).real() - GetLocalOprExp("Sy2", site).real() );
        //     ObsOutFile2<<site<<"  "<<SS<<"  "<<TzTz<<endl;
        // }




    } //Temperature loop

} // ---------


#endif // MCENGINE_H
