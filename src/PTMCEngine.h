#include <math.h>
#include "ParametersEngine.h"
#include "tensor_type.h"
#include "random"
#include <stdlib.h>
#define PI acos(-1.0)

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef PTMCENGINE_H
#define PTMCENGINE_H

class PTMCEngine
{
public:
    PTMCEngine(Parameters &Parameters__, mt19937_64 &Generator1__)
        : Parameters_(Parameters__), Generator1_(Generator1__),
        ns_(Parameters_.ns)
    {
    }



    void InitializeEngine();
    void Create_Connections_wrt_site();
    double random1();
    void Initialize_MarsagliaParams();
    double Lorentzian(double x, double brd);
    void RUN_MC();
    double Get_Local_Energy(int site, int Ti);
    void FieldThrow(int site, int Ti);
    complex<double> GetLocalOprExp(string opr_str, int opr_site, int Ti);
    double Calculate_TotalE(int Ti);
    void TotalE_Average(int Replica_no, int Confs_used);
    void Calculate_SiSj(int Ti);
    void Calculate_TauZiTauZj(int Ti);
    void TauZiTauZj_Average(int Confs_used);
    void SiSj_Average(int Confs_used);

    complex<double> GetLocalOprExpAnsatz(string opr_str, int Site, string StateType);
    double Calculate_TotalE_Ansatz(string StateType);
    void Run_Ansatz(string StateType);

    void Tag_Sites();
    void Initialize_O3_Params();


    bool PTMC; //Parallel Tempering Monte Carlo
    int N_replica_sets;
    int N_temperature_slices;
    int NProcessors;
    int SwappingSweepGap;
    Mat_2_doub TemperatureReplicaSets;

    double TotalE_Mean_;
    double TotalE_square_Mean_;

    int N_Phi, N_Theta;
    double d_Phi, d_Theta;
    Mat_1_doub Distribution_Phi1, Distribution_Phi2, Distribution_Phi3;
    Mat_1_doub Distribution_Theta1, Distribution_Theta2;

    Mat_1_int AccCount;
    double WindowSize, AccRatio;

    Mat_2_int LocalConnectionSize;
    Mat_3_int LocalConnectionSites;
    Mat_3_string LocalConnectionOprs;
    Mat_2_doub LocalConnectionValue;

    int theta_O3_ind, phi_O3_ind;
    Mat_2_doub theta_O3, phi_O3;
    Mat_1_string SiteTags;
    Mat_1_doub SiteSpinVal;
    int theta1_ind, theta2_ind, phi1_ind, phi2_ind, phi3_ind;
    int X_, Y_, Z_;
    Mat_2_Complex_doub Dvecs;
    Mat_2_doub theta1_, theta2_, phi1_, phi2_, phi3_;
    Parameters &Parameters_;
    mt19937_64 &Generator1_; //for random fields
    const int ns_;


    //double TotalE_, TotalE_Mean_, TotalE_square_Mean_;
    Matrix<double> SiSj_, SiSj_Mean_, SiSj_square_Mean_;
    Matrix<double> TauZiTauZj_, TauZiTauZj_Mean_, TauZiTauZj_square_Mean_;

    uniform_real_distribution<double> dis1_; //for random fields

};

/*
 * ***********
 *  Functions in Class MCEngine ------
 *  ***********
*/


void PTMCEngine::Initialize_O3_Params(){

    // Dvecs.resize(ns_);
    // for(int i=0;i<ns_;i++){
    //     Dvecs[i].resize(3);
    // }

    theta_O3_ind=0;
    phi_O3_ind=1;

    for(int Ti=0;Ti<N_temperature_slices;Ti++){
        for(int i=0;i<ns_;i++){
            theta_O3[Ti][i] = PI*random1();
            phi_O3[Ti][i] = 2.0*PI*random1();
        }
    }

    // Update_Dvecs();

}

void PTMCEngine::Tag_Sites(){

    SiteTags.resize(ns_);
    SiteSpinVal.resize(ns_);

    ifstream FileTags(Parameters_.File_SiteTags.c_str());
    int site_temp;
    for(int i=0;i<ns_;i++){
        FileTags>>site_temp>>SiteTags[i]>>SiteSpinVal[i];
    }

}


void PTMCEngine::InitializeEngine(){

    PTMC=Parameters_.PTMC;
    N_replica_sets=Parameters_.N_replica_sets;
    N_temperature_slices=Parameters_.N_temperature_slices;
    NProcessors=Parameters_.NProcessors;
    SwappingSweepGap=Parameters_.SwappingSweepGap;
    TemperatureReplicaSets=Parameters_.TemperatureReplicaSets;

    theta1_.resize(N_temperature_slices);theta2_.resize(N_temperature_slices);
    phi1_.resize(N_temperature_slices);phi2_.resize(N_temperature_slices);phi3_.resize(N_temperature_slices);

    for(int Ti=0;Ti<N_temperature_slices;Ti++){
        theta1_[Ti].resize(ns_);theta2_[Ti].resize(ns_);
        phi1_[Ti].resize(ns_);phi2_[Ti].resize(ns_);phi3_[Ti].resize(ns_);
    }


    theta_O3.resize(N_temperature_slices);
    phi_O3.resize(N_temperature_slices);
    for(int Ti=0;Ti<N_temperature_slices;Ti++){
        theta_O3[Ti].resize(ns_);
        phi_O3[Ti].resize(ns_);
    }

    Tag_Sites();
    Initialize_MarsagliaParams();
    Initialize_O3_Params();

}


double PTMCEngine::Calculate_TotalE_Ansatz(string StateType){


    complex<double> temp_TotalE_=0.0;
    double TotalE_=0.0;

    for(int FileNo=0;FileNo<Parameters_.ConnectionFiles.size();FileNo++){

        Mat_1_Complex_doub E_array;
        int Np_=1;

#ifdef _OPENMP
        Np_= omp_get_max_threads();
#endif

        E_array.resize(Np_);

#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
        for(int connection_no=0;connection_no<Parameters_.Connections[FileNo].size();connection_no++){

            int thread_id=0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif


            stringstream connection_stream;
            connection_stream<<Parameters_.Connections[FileNo][connection_no];

            int n_oprs;
            string temp_opr_str;
            Mat_1_string oprs_list;
            Mat_1_int oprs_site;
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
                E_conn = E_conn*GetLocalOprExpAnsatz(opr_str, opr_site, StateType);
            }
            E_conn = E_conn*connection_val;


            E_array[thread_id] += E_conn;

        }

        for(int th_id=0;th_id<Np_;th_id++){
            temp_TotalE_ +=E_array[th_id];
        }

    }



    if(temp_TotalE_.imag()>0.000001){
        cout<<"temp_TotalE_.real() : "<<temp_TotalE_.real()<<endl;
        cout<<"temp_TotalE_.imag() : "<<temp_TotalE_.imag()<<endl;
        assert(temp_TotalE_.imag()<0.000001);
    }

    TotalE_ =  temp_TotalE_.real();

    return TotalE_;

}




double PTMCEngine::Calculate_TotalE(int Ti){


    complex<double> temp_TotalE_=0.0;
    double TotalE_=0.0;

    for(int FileNo=0;FileNo<Parameters_.ConnectionFiles.size();FileNo++){

        Mat_1_Complex_doub E_array;
        int Np_=1;

#ifdef _OPENMP
        Np_= omp_get_max_threads();
#endif

        E_array.resize(Np_);

#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
        for(int connection_no=0;connection_no<Parameters_.Connections[FileNo].size();connection_no++){

            int thread_id=0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif


            stringstream connection_stream;
            connection_stream<<Parameters_.Connections[FileNo][connection_no];

            int n_oprs;
            string temp_opr_str;
            Mat_1_string oprs_list;
            Mat_1_int oprs_site;
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
                E_conn = E_conn*GetLocalOprExp(opr_str, opr_site, Ti);
            }
            E_conn = E_conn*connection_val;


            E_array[thread_id] += E_conn;

        }

        for(int th_id=0;th_id<Np_;th_id++){
            temp_TotalE_ +=E_array[th_id];
        }

    }



    if(temp_TotalE_.imag()>0.000001){
        cout<<"temp_TotalE_.real() : "<<temp_TotalE_.real()<<endl;
        cout<<"temp_TotalE_.imag() : "<<temp_TotalE_.imag()<<endl;
        assert(temp_TotalE_.imag()<0.000001);
    }

    TotalE_ =  temp_TotalE_.real();

    return TotalE_;

}


complex<double> PTMCEngine::GetLocalOprExpAnsatz(string opr_str, int Site, string StateType){

    complex<double> value;

    double x1, x2, y1, y2, z1, z2;
    complex<double> x_,y_,z_;

    if(StateType=="AFM_AFQ"){

        //Assuming square lattice of S and L
        int sitex, sitey;
        int lx,ly;
        lx=int(sqrt(0.5*ns_)+0.5);
        ly=(ns_/(2*lx));

        if(Site<(ns_/2)){
            sitex = Site%lx;
            sitey = Site/lx;
        }
        else{
            sitex = (Site - (ns_/2))%lx;
            sitey = (Site - (ns_/2))/lx;
        }

        string SiteType;
        if(Site<(ns_/2)){
            SiteType="S_";
        }
        else{
            SiteType="L_";
        }
        if((sitex+sitey)%2==0){
            SiteType +="A";
        }
        else{
            SiteType +="B";
        }


        if(SiteType=="S_A"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="S_B"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=-1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_A"){
            x1=1.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_B"){
            x1=0.0; x2=0.0;
            y1=1.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else{
            cout<<"SiteType not allowed in this Ansatz"<<endl;
            assert(false);
        }

    }


    if(StateType=="AFM_FQ"){

        //Assuming square lattice of S and L
        int sitex, sitey;
        int lx,ly;
        lx=int(sqrt(0.5*ns_)+0.5);
        ly=(ns_/(2*lx));

        if(Site<(ns_/2)){
            sitex = Site%lx;
            sitey = Site/lx;
        }
        else{
            sitex = (Site - (ns_/2))%lx;
            sitey = (Site - (ns_/2))/lx;
        }

        string SiteType;
        if(Site<(ns_/2)){
            SiteType="S_";
        }
        else{
            SiteType="L_";
        }
        if((sitex+sitey)%2==0){
            SiteType +="A";
        }
        else{
            SiteType +="B";
        }


        if(SiteType=="S_A"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="S_B"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=-1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_A"){
            x1=1.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_B"){
            // x1=0.0; x2=0.0;
            // y1=1.0; y2=0.0;
            // z1=0.0;z2=0.0;
            x1=1.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else{
            cout<<"SiteType not allowed in this Ansatz"<<endl;
            assert(false);
        }

    }

    if(StateType=="FM_AFQ"){

        //Assuming square lattice of S and L
        int sitex, sitey;
        int lx,ly;
        lx=int(sqrt(0.5*ns_)+0.5);
        ly=(ns_/(2*lx));

        if(Site<(ns_/2)){
            sitex = Site%lx;
            sitey = Site/lx;
        }
        else{
            sitex = (Site - (ns_/2))%lx;
            sitey = (Site - (ns_/2))/lx;
        }

        string SiteType;
        if(Site<(ns_/2)){
            SiteType="S_";
        }
        else{
            SiteType="L_";
        }
        if((sitex+sitey)%2==0){
            SiteType +="A";
        }
        else{
            SiteType +="B";
        }


        if(SiteType=="S_A"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="S_B"){
            // x1=1.0/sqrt(2.0); x2=0.0;
            // y1=0.0; y2=-1.0/sqrt(2.0);
            // z1=0.0;z2=0.0;
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_A"){
            x1=1.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_B"){
            x1=0.0; x2=0.0;
            y1=1.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else{
            cout<<"SiteType not allowed in this Ansatz"<<endl;
            assert(false);
        }

    }

    if(StateType=="FM_FQ"){

        //Assuming square lattice of S and L
        int sitex, sitey;
        int lx,ly;
        lx=int(sqrt(0.5*ns_)+0.5);
        ly=(ns_/(2*lx));

        if(Site<(ns_/2)){
            sitex = Site%lx;
            sitey = Site/lx;
        }
        else{
            sitex = (Site - (ns_/2))%lx;
            sitey = (Site - (ns_/2))/lx;
        }

        string SiteType;
        if(Site<(ns_/2)){
            SiteType="S_";
        }
        else{
            SiteType="L_";
        }
        if((sitex+sitey)%2==0){
            SiteType +="A";
        }
        else{
            SiteType +="B";
        }


        if(SiteType=="S_A"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="S_B"){
            // x1=1.0/sqrt(2.0); x2=0.0;
            // y1=0.0; y2=-1.0/sqrt(2.0);
            // z1=0.0;z2=0.0;
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_A"){
            x1=1.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_B"){
            // x1=0.0; x2=0.0;
            // y1=1.0; y2=0.0;
            // z1=0.0;z2=0.0;
            x1=1.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=0.0;z2=0.0;
        }
        else{
            cout<<"SiteType not allowed in this Ansatz"<<endl;
            assert(false);
        }

    }

    if(StateType=="FM_AFQ_pZmXY"){

        //Assuming square lattice of S and L
        int sitex, sitey;
        int lx,ly;
        lx=int(sqrt(0.5*ns_)+0.5);
        ly=(ns_/(2*lx));

        if(Site<(ns_/2)){
            sitex = Site%lx;
            sitey = Site/lx;
        }
        else{
            sitex = (Site - (ns_/2))%lx;
            sitey = (Site - (ns_/2))/lx;
        }

        string SiteType;
        if(Site<(ns_/2)){
            SiteType="S_";
        }
        else{
            SiteType="L_";
        }
        if((sitex+sitey)%2==0){
            SiteType +="A";
        }
        else{
            SiteType +="B";
        }


        if(SiteType=="S_A"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="S_B"){
            // x1=1.0/sqrt(2.0); x2=0.0;
            // y1=0.0; y2=-1.0/sqrt(2.0);
            // z1=0.0;z2=0.0;
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=0.0; y2=1.0/sqrt(2.0);
            z1=0.0;z2=0.0;
        }
        else if(SiteType=="L_A"){
            x1=0.0; x2=0.0;
            y1=0.0; y2=0.0;
            z1=1.0;z2=0.0;
        }
        else if(SiteType=="L_B"){
            x1=1.0/sqrt(2.0); x2=0.0;
            y1=1.0/sqrt(2.0); y2=0.0;
            z1=0.0;z2=0.0;
        }
        else{
            cout<<"SiteType not allowed in this Ansatz"<<endl;
            assert(false);
        }

    }



    x_=complex<double>(x1,x2);
    y_=complex<double>(y1,y2);
    z_=complex<double>(z1,z2);

    if(opr_str == "d1"){
        value = x_;
    }
    if(opr_str == "d2"){
        value = y_;
    }
    if(opr_str == "d3"){
        value = z_;
    }
    if(opr_str == "Sz"){
        value = iota_complex*( x_*conj(y_) - y_*conj(x_) );
    }
    if(opr_str == "Sx"){
        value = iota_complex*( y_*conj(z_) - z_*conj(y_) );
    }
    if(opr_str == "Sy"){
        value = iota_complex*( z_*conj(x_) - x_*conj(z_) );
    }
    if(opr_str == "Sp"){
        value = iota_complex*( y_*conj(z_) - z_*conj(y_) ) - 1.0*( z_*conj(x_) - x_*conj(z_) );
    }
    if(opr_str == "Sm"){
        value = iota_complex*( y_*conj(z_) - z_*conj(y_) ) + 1.0*( z_*conj(x_) - x_*conj(z_) );
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




complex<double> PTMCEngine::GetLocalOprExp(string opr_str, int opr_site, int Ti){

    complex<double> value;


    if(SiteTags[opr_site]=="CP2"){

        double x1, x2, y1, y2, z1, z2;
        complex<double> x_,y_,z_;
        x1 = sqrt(sqrt(theta2_[Ti][opr_site])) * sqrt(theta1_[Ti][opr_site]) * sin(phi1_[Ti][opr_site]);
        x2 = sqrt(sqrt(theta2_[Ti][opr_site])) * sqrt(theta1_[Ti][opr_site]) * cos(phi1_[Ti][opr_site]);

        y1 = sqrt(sqrt(theta2_[Ti][opr_site])) * sqrt(1.0 - theta1_[Ti][opr_site]) * sin(phi2_[Ti][opr_site]);
        y2 = sqrt(sqrt(theta2_[Ti][opr_site])) * sqrt(1.0 - theta1_[Ti][opr_site]) * cos(phi2_[Ti][opr_site]);

        z1 = sqrt(1.0 - sqrt(theta2_[Ti][opr_site])) * sin(phi3_[Ti][opr_site]);
        z2 = sqrt(1.0 - sqrt(theta2_[Ti][opr_site])) * cos(phi3_[Ti][opr_site]);


        x_=complex<double>(x1,x2);
        y_=complex<double>(y1,y2);
        z_=complex<double>(z1,z2);

        if(opr_str == "d1"){
            value = x_;
        }
        if(opr_str == "d2"){
            value = y_;
        }
        if(opr_str == "d3"){
            value = z_;
        }
        if(opr_str == "Sz"){
            value = iota_complex*( x_*conj(y_) - y_*conj(x_) );
        }
        if(opr_str == "Sx"){
            value = iota_complex*( y_*conj(z_) - z_*conj(y_) );
        }
        if(opr_str == "Sy"){
            value = iota_complex*( z_*conj(x_) - x_*conj(z_) );
        }
        if(opr_str == "Sp"){
            value = iota_complex*( y_*conj(z_) - z_*conj(y_) ) - 1.0*( z_*conj(x_) - x_*conj(z_) );
        }
        if(opr_str == "Sm"){
            value = iota_complex*( y_*conj(z_) - z_*conj(y_) ) + 1.0*( z_*conj(x_) - x_*conj(z_) );
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


    }
    else if(SiteTags[opr_site]=="O3"){

        if(opr_str == "Sz"){
            value = SiteSpinVal[opr_site]*cos(theta_O3[Ti][opr_site]);
        }
        else if(opr_str == "Sx"){
            value = SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*cos(phi_O3[Ti][opr_site]);
        }
        else if(opr_str == "Sy"){
            value = SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*sin(phi_O3[Ti][opr_site]);
        }
        else if(opr_str == "Sp"){
            value = SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*exp(iota_complex*(phi_O3[Ti][opr_site]));
        }
        else if(opr_str == "Sm"){
            value = SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*exp(-1.0*iota_complex*(phi_O3[Ti][opr_site]));
        }
        else if(opr_str == "Sx2"){
            value = SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*cos(phi_O3[Ti][opr_site])*
                    SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*cos(phi_O3[Ti][opr_site]);
        }
        else if(opr_str == "Sy2"){
            value = SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*sin(phi_O3[Ti][opr_site])*
                    SiteSpinVal[opr_site]*sin(theta_O3[Ti][opr_site])*sin(phi_O3[Ti][opr_site]);
        }
        else if(opr_str == "Sz2"){
            value = SiteSpinVal[opr_site]*cos(theta_O3[Ti][opr_site])*SiteSpinVal[opr_site]*cos(theta_O3[Ti][opr_site]);
        }
        else{
            cout<<"Only Sz, Sx, Sy, Sp, Sm, Sx2, Sy2, Sz2 oprs allowed in O3 MonteCarlo"<<endl;
            assert(false);
        }

    }
    else{
        assert(false);
    }

    return value;

}


void PTMCEngine::Create_Connections_wrt_site(){

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



double PTMCEngine::random1()
{
    return dis1_(Generator1_);
}

void PTMCEngine::Initialize_MarsagliaParams(){

    X_=0;Y_=1;Z_=2;
    theta1_ind=0;theta2_ind=1;
    phi1_ind=2;phi2_ind=3;phi3_ind=4;


    // Dvecs.resize(ns_);
    // for(int i=0;i<ns_;i++){
    //     Dvecs[i].resize(3);
    // }


    for(int Ti=0;Ti<N_temperature_slices;Ti++){
        for(int i=0;i<ns_;i++){
            theta1_[Ti][i] = random1();
            theta2_[Ti][i] = random1();
            phi1_[Ti][i] = 2.0*PI*random1();
            phi2_[Ti][i] = 2.0*PI*random1();
            //phi3_[Ti][i] = 2.0*PI*random1();
            phi3_[Ti][i] = 0.5*PI;
        }
    }

    // Update_Dvecs();

}


double PTMCEngine::Lorentzian(double x, double brd)
{
    double temp;

    temp = (1.0 / PI) * ((brd / 2.0) / ((x * x) + ((brd * brd) / 4.0)));

    return temp;
}

double PTMCEngine::Get_Local_Energy(int site, int Ti){

    complex<double> E_local=0;
    Mat_1_Complex_doub E_local_array;
    int Np_=1;

#ifdef _OPENMP
    Np_= omp_get_max_threads();
#endif

    E_local_array.resize(Np_);

#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
    for(int conn_no=0;conn_no<LocalConnectionSize[site].size();conn_no++){
        int thread_id=0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif

        complex<double> E_conn=1.0;
        for(int opr_no=0;opr_no<LocalConnectionSize[site][conn_no];opr_no++){
            int opr_site = LocalConnectionSites[site][conn_no][opr_no];
            string opr_str = LocalConnectionOprs[site][conn_no][opr_no];
            E_conn = E_conn*GetLocalOprExp(opr_str, opr_site, Ti);
        }
        E_conn = E_conn*LocalConnectionValue[site][conn_no];

        E_local_array[thread_id] += E_conn;

    }


    for(int th_i=0;th_i<Np_;th_i++){
        E_local += E_local_array[th_i];
    }

    if(E_local.imag()>0.00001){
        cout<<"E_local.imag() = "<<E_local.imag()<<endl;
        assert(E_local.imag()<0.00001);
    }

    return E_local.real();
}

void PTMCEngine::FieldThrow(int site, int Ti)
{

    /*
    theta1_[site] = random1();
    theta2_[site] = random1();
    phi1_[site] = 2.0*PI*random1();
    phi2_[site] = 2.0*PI*random1();
    phi3_[site] = 2.0*PI*random1();
*/


    if(SiteTags[site]=="CP2"){
        theta1_[Ti][site] += WindowSize*( random1() -0.5 );
        theta2_[Ti][site] += WindowSize*( random1() -0.5 );
        phi1_[Ti][site] += 2.0*PI*WindowSize*( random1() -0.5 );
        phi2_[Ti][site] += 2.0*PI*WindowSize*( random1() -0.5 );
        //phi3_[Ti][site] += 2.0*PI*WindowSize*( random1() -0.5 );
        phi3_[Ti][site] = 0.5*PI;

        // theta1_[site] += 2.0*(random1()-0.5);
        // theta2_[site] += 2.0*(random1()-0.5);
        // phi1_[site] += 2.0*PI*2.0*(random1()-0.5);
        // phi2_[site] += 2.0*PI*2.0*(random1()-0.5);
        // phi3_[site] += 2.0*PI*2.0*(random1()-0.5);


        if (phi1_[Ti][site] < 0.0){phi1_[Ti][site] = 2.0 * PI + phi1_[Ti][site];}
        phi1_[Ti][site] = fmod(phi1_[Ti][site], 2.0 * PI);

        if (phi2_[Ti][site] < 0.0){phi2_[Ti][site] =  2.0 * PI + phi2_[Ti][site];}
        phi2_[Ti][site] = fmod(phi2_[Ti][site], 2.0 * PI);

        if (phi3_[Ti][site] < 0.0){phi3_[Ti][site] =  2.0 * PI + phi3_[Ti][site];}
        phi3_[Ti][site] = fmod(phi3_[Ti][site], 2.0 * PI);


        if (theta1_[Ti][site] < 0.0){theta1_[Ti][site] =  1.0 + theta1_[Ti][site];}
        theta1_[Ti][site] = fmod(theta1_[Ti][site], 1.0);

        if (theta2_[Ti][site] < 0.0){theta2_[Ti][site] =  1.0 + theta2_[Ti][site];}
        theta2_[Ti][site] = fmod(theta2_[Ti][site], 1.0);
    }
    else if(SiteTags[site]=="O3"){

        phi_O3[Ti][site] += 2 * PI * (random1() - 0.5) * WindowSize;
        if( phi_O3[Ti][site] < 0.0) {phi_O3[Ti][site] += 2.0*PI; }
        if( phi_O3[Ti][site] >=2.0*PI) {phi_O3[Ti][site] -= 2.0*PI;}


        theta_O3[Ti][site] += PI * (random1() - 0.5) * WindowSize;
        if ( theta_O3[Ti][site] < 0.0 ) {
            theta_O3[Ti][site] = -1.0*theta_O3[Ti][site];
            phi_O3[Ti][site] = fmod( phi_O3[Ti][site]+PI, 2.0*PI );
        }
        if ( theta_O3[Ti][site] > PI ) {
            theta_O3[Ti][site] = 2.0*PI - theta_O3[Ti][site];
            phi_O3[Ti][site] = fmod( phi_O3[Ti][site] + PI, 2.0*PI );
        }

    }
    else{
        cout<<"SiteTag is not correct"<<endl;
        assert(false);
    }



} // ----------



void PTMCEngine::Calculate_SiSj(int Ti){

#ifdef _OPENMPX
#pragma omp parallel for default(shared)
#endif
    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            SiSj_(site_i, site_j) =  GetLocalOprExp("Sx", site_i, Ti).real()*GetLocalOprExp("Sx", site_j, Ti).real()
            + GetLocalOprExp("Sy", site_i, Ti).real()*GetLocalOprExp("Sy", site_j, Ti).real()
                + GetLocalOprExp("Sz", site_i, Ti).real()*GetLocalOprExp("Sz", site_j, Ti).real();
        }
    }

}


void PTMCEngine::Calculate_TauZiTauZj(int Ti){

#ifdef _OPENMPX
#pragma omp parallel for default(shared)
#endif
    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            TauZiTauZj_(site_i, site_j) =  0.25*( GetLocalOprExp("Sx2", site_i, Ti).real() - GetLocalOprExp("Sy2", site_i, Ti).real() )*
                                          ( GetLocalOprExp("Sx2", site_j, Ti).real() - GetLocalOprExp("Sy2", site_j, Ti).real() );
        }
    }

}


void PTMCEngine::TauZiTauZj_Average(int Confs_used)
{
#ifdef _OPENMPX
#pragma omp parallel for default(shared)
#endif
    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            TauZiTauZj_Mean_(site_i, site_j)  = ((TauZiTauZj_Mean_(site_i, site_j)*(Confs_used-1)) + TauZiTauZj_(site_i, site_j) )/(1.0*Confs_used);
            TauZiTauZj_square_Mean_(site_i, site_j) = ((TauZiTauZj_square_Mean_(site_i, site_j)*(Confs_used-1)) + (TauZiTauZj_(site_i, site_j)*TauZiTauZj_(site_i, site_j)) )/(1.0*Confs_used);

            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------



void PTMCEngine::SiSj_Average(int Confs_used)
{
#ifdef _OPENMPX
#pragma omp parallel for default(shared)
#endif
    for (int site_i = 0; site_i < ns_; site_i++)
    {
        for (int site_j = 0; site_j <ns_; site_j++)
        {
            SiSj_Mean_(site_i, site_j)  = ((SiSj_Mean_(site_i, site_j)*(Confs_used-1)) + SiSj_(site_i, site_j) )/(1.0*Confs_used);
            SiSj_square_Mean_(site_i, site_j) = ((SiSj_square_Mean_(site_i, site_j)*(Confs_used-1)) + (SiSj_(site_i, site_j)*SiSj_(site_i, site_j)) )/(1.0*Confs_used);

            //cout << qx << " "<< qy<< " "<<  SiSjQ_(qx,qy) << endl;
        }
    }

} // ----------



void PTMCEngine::TotalE_Average(int Replica_no, int Confs_used){

    double TotalE_ = Calculate_TotalE(Replica_no);


    TotalE_Mean_ = ((TotalE_Mean_*(Confs_used-1)) + TotalE_ )/(1.0*Confs_used);
    TotalE_square_Mean_ = ( (TotalE_square_Mean_*(Confs_used-1))  +  (TotalE_*TotalE_) )/ ( 1.0*Confs_used );

}

void PTMCEngine::Run_Ansatz(string StateType){


    cout<<"Energy "<<StateType<<" = "<<Calculate_TotalE_Ansatz(StateType)<<endl;


    string ObsOutFile_str="MicrostateLocalObs_Ansatz_"+StateType+".txt";
    ofstream ObsOutFile(ObsOutFile_str.c_str());
    ObsOutFile<<"#site Sx Sy Sz Sx2 Sy2 Sz2  d1  d2  d3"<<endl;
    for(int site=0;site<ns_;site++){
        ObsOutFile<<site<<setw(15)<<GetLocalOprExpAnsatz("Sx", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("Sx", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("Sy", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("Sy", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("Sz", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("Sz", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("Sx2", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("Sx2", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("Sy2", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("Sy2", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("Sz2", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("Sz2", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("d1", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("d1", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("d2", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("d2", site, StateType).imag()<<setw(15)
        <<GetLocalOprExpAnsatz("d3", site, StateType).real()<<setw(15)<<GetLocalOprExpAnsatz("d3", site, StateType).imag()
        <<endl;
    }

}

void PTMCEngine::RUN_MC()
{


    SiSj_.resize(ns_,ns_);SiSj_Mean_.resize(ns_,ns_);SiSj_square_Mean_.resize(ns_,ns_);
    TauZiTauZj_.resize(ns_,ns_);TauZiTauZj_Mean_.resize(ns_,ns_);TauZiTauZj_square_Mean_.resize(ns_,ns_);


    AccCount.resize(2);


    int MC_sweeps_used_for_Avg = Parameters_.Last_n_sweeps_for_measurement;
    int Gap_bw_sweeps = Parameters_.Measurement_after_each_m_sweeps;

    double prob_swap;

    WindowSize=0.2;
    int Lowest_Temp_index=0;


    Mat_1_int No_of_swaps;
    No_of_swaps.resize(N_temperature_slices-1);
    //	assert(false);



    string ReplicaExchangeoutstr="ReplicaExchange.txt";
    ofstream ReplicaExchangeout(ReplicaExchangeoutstr.c_str());

    int RoundTrips=0;
    int TouchedReference=0;
    int Reference_slice=N_temperature_slices-1;

    for(int temp_set=0;temp_set<N_replica_sets;temp_set++){

        int swapping_set=0;
        int swapping_type=0;
        TotalE_Mean_=0.0;
        TotalE_square_Mean_=0.0;
        int measure_start=0;
        int Confs_used=0;
        Mat_1_int Replica_Permutation;
        Replica_Permutation.resize(N_temperature_slices);
        for(int Ti=0;Ti<N_temperature_slices;Ti++){
            Replica_Permutation[Ti]=Ti;
        }

        for(int sweep_no=0;sweep_no<Parameters_.IterMax;sweep_no++){

            if(N_temperature_slices>1){
                if(sweep_no%(1*SwappingSweepGap)==0){
                    ReplicaExchangeout<<sweep_no;
                    for(int Ti=0;Ti<N_temperature_slices;Ti++){
                        ReplicaExchangeout<<setw(15)<<Replica_Permutation[Ti];
                    }
                    ReplicaExchangeout<<endl;

                    cout<<"Round Trips : "<<RoundTrips<<endl;
                }
            }

#ifdef _OPENMPX
#pragma omp parallel for default(shared)
#endif
            for(int Ti=0;Ti<N_temperature_slices;Ti++){
                int thread_id;
#ifdef _OPENMPX
                thread_id = omp_get_thread_num();
#endif

                double beta = double( 1.0/(Parameters_.Boltzman_constant*TemperatureReplicaSets[temp_set][Ti]));

                double LocalPrevE, LocalCurrE, P12, Prob_check;
                double saved_Params[5];
                for (int i = 0; i < ns_; i++)
                { //For each site


                    LocalPrevE = Get_Local_Energy(i, Ti);

                    if(SiteTags[i]=="CP2"){
                        saved_Params[theta1_ind] = theta1_[Ti][i];
                        saved_Params[theta2_ind] = theta2_[Ti][i];
                        saved_Params[phi1_ind] = phi1_[Ti][i];
                        saved_Params[phi2_ind] = phi2_[Ti][i];
                        saved_Params[phi3_ind] = phi3_[Ti][i];
                    }
                    else if(SiteTags[i]=="O3"){
                        saved_Params[theta_O3_ind] = theta_O3[Ti][i];
                        saved_Params[phi_O3_ind] = phi_O3[Ti][i];
                    }
                    else{
                        cout<<"SiteTag Not correct"<<endl;
                        assert(false);
                    }

                    FieldThrow(i, Ti);

                    //cout<<"Change for thread = "<<thread_id<<" : "<<theta1_[Ti][i]-saved_Params[theta1_ind]<<endl;

                    LocalCurrE = Get_Local_Energy(i, Ti);

                    P12 = beta * ((LocalPrevE) - (LocalCurrE));

                    if(SiteTags[i]=="O3"){
                    P12 += log((sin(theta_O3[Ti][i]) / sin(saved_Params[theta_O3_ind])));
                    }
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
                    if (Prob_check > (random1()) ) //random1()
                    {
                        //PrevE = CurrE;
                        //AccCount[0] +=1;
                    }
                    //REJECTED
                    else
                    {
                        //AccCount[1] +=1;
                        if(SiteTags[i]=="CP2"){
                            theta1_[Ti][i] = saved_Params[theta1_ind];
                            theta2_[Ti][i] = saved_Params[theta2_ind];
                            phi1_[Ti][i] = saved_Params[phi1_ind];
                            phi2_[Ti][i] = saved_Params[phi2_ind];
                            phi3_[Ti][i] = saved_Params[phi3_ind];
                        }
                        if(SiteTags[i]=="O3"){
                            theta_O3[Ti][i] = saved_Params[theta_O3_ind];
                            phi_O3[Ti][i] = saved_Params[phi_O3_ind];
                        }
                    }

                } // site loop

            } //Replica no. i.e. Ti


            if(N_temperature_slices>1){
                if(sweep_no%SwappingSweepGap==0){

                    // if(swapping_set==(N_temperature_slices-1)){
                    //     swapping_set=0;
                    // }

                    int T_lowest;
                    T_lowest=swapping_type;

                    for(int Ti_temp_original=T_lowest;Ti_temp_original<(N_temperature_slices-1);Ti_temp_original=Ti_temp_original+2){
                        int Tj_temp_original=Ti_temp_original+1;


                        int Ti_temp = Replica_Permutation[Ti_temp_original];
                        int Tj_temp = Replica_Permutation[Tj_temp_original];

                        double Bi=1.0/(Parameters_.Boltzman_constant*TemperatureReplicaSets[temp_set][Ti_temp]);
                        double Bj=1.0/(Parameters_.Boltzman_constant*TemperatureReplicaSets[temp_set][Tj_temp]);
                        double Ei=Calculate_TotalE(Ti_temp);
                        double Ej=Calculate_TotalE(Tj_temp);

                        if(sweep_no%1000==0){
                            cout<<sweep_no<<" : "<<TemperatureReplicaSets[temp_set][Ti_temp]<<"  "<<Ei<<endl;
                            cout<<sweep_no<<" : "<<TemperatureReplicaSets[temp_set][Tj_temp]<<"  "<<Ej<<endl;
                        }

                        prob_swap = min(1.0, exp( (Bi-Bj)*(Ei-Ej) )  );

                        if(prob_swap > random1()){
                            double Temp_j_temp = TemperatureReplicaSets[temp_set][Tj_temp];
                            TemperatureReplicaSets[temp_set][Tj_temp]=TemperatureReplicaSets[temp_set][Ti_temp];
                            TemperatureReplicaSets[temp_set][Ti_temp]=Temp_j_temp;

                            Replica_Permutation[Ti_temp_original] = Tj_temp;
                            Replica_Permutation[Tj_temp_original] = Ti_temp;

                            No_of_swaps[Ti_temp_original] +=1;
                        }
                    }

                    swapping_type = (swapping_type+1)%2;

                }
            }
            else{
                double Ei=Calculate_TotalE(0);
                if(sweep_no%1000==0){
                    cout<<sweep_no<<" : "<<TemperatureReplicaSets[temp_set][Replica_Permutation[0]]<<"  "<<Ei<<endl;
                }
            }






            //in measurement loops
            if(sweep_no > (Parameters_.IterMax - (Gap_bw_sweeps * (MC_sweeps_used_for_Avg - 1) + MC_sweeps_used_for_Avg))){

                if (measure_start == 0)
                {
                    measure_start++;
                    cout << "----------Measurement is started----------" << endl;
                }
                int temp_count = sweep_no -
                                 (Parameters_.IterMax - (Gap_bw_sweeps * (MC_sweeps_used_for_Avg - 1) + MC_sweeps_used_for_Avg));
                int zero_or_not = temp_count % (Gap_bw_sweeps + 1);
                if (zero_or_not == 0)
                {

                    Confs_used = Confs_used + 1;

                    Calculate_SiSj(Replica_Permutation[0]);
                    SiSj_Average(Confs_used);
                    Calculate_TauZiTauZj(Replica_Permutation[0]);
                    TauZiTauZj_Average(Confs_used);

                    TotalE_Average(Replica_Permutation[0] , Confs_used);
                    if(Confs_used%500==0){
                        double Temp_ = TemperatureReplicaSets[temp_set][Replica_Permutation[0]];
                        double std_E = max(0.0,(((TotalE_square_Mean_) - ((TotalE_Mean_ * TotalE_Mean_)))));
                        //double std_E = ((((TotalE_square_Mean_ / (Confs_used * 1.0)) - ((TotalE_Mean_ * TotalE_Mean_) / (Confs_used * Confs_used * 1.0)))));
                        //cout<<Confs_used<<" : Temperature, E2_mean :"<<setw(15)<<Temp_<<setw(15)<<TotalE_square_Mean_ / (Confs_used * 1.0)<<endl;
                        //cout<<Confs_used<<" : Temperature, E_mean :"<<setw(15)<<Temp_<<setw(15)<<TotalE_Mean_ / (Confs_used * 1.0)<<endl;

                        cout<<Confs_used<<" : Temperature, Energy, std(Energy) : "<<setw(15)<<Temp_<<setw(15)<<(TotalE_Mean_)<<setw(15)<<std_E<<endl;
                        cout<<Confs_used<<" : Temperature, Cv :"<<setw(15)<<Temp_<<setw(15)<<std_E/(Parameters_.Boltzman_constant*Parameters_.Boltzman_constant*Temp_*Temp_)<<endl;
                    }

                }

                /*
            if(Confs_used%500==0){
                double Temp_ = TemperatureReplicaSets[temp_set][Lowest_Temp_index];
                double std_E = max(0.0,(((TotalE_square_Mean_ / (Confs_used * 1.0)) - ((TotalE_Mean_ * TotalE_Mean_) / (Confs_used * Confs_used * 1.0)))));
                //double std_E = ((((TotalE_square_Mean_ / (Confs_used * 1.0)) - ((TotalE_Mean_ * TotalE_Mean_) / (Confs_used * Confs_used * 1.0)))));
                 cout<<Confs_used<<" : Temperature, Energy, std(Energy) : "<<setw(15)<<Temp_<<setw(15)<<(TotalE_Mean_/Confs_used)<<setw(15)<<std_E<<endl;
                cout<<Confs_used<<" : Temperature, Cv :"<<setw(15)<<Temp_<<setw(15)<<std_E/(Parameters_.Boltzman_constant*Parameters_.Boltzman_constant*Temp_*Temp_)<<endl;
            }
            */


            }


            if(Replica_Permutation[0]==Reference_slice){
                TouchedReference=1;
            }

            if(Replica_Permutation[0]==0 && TouchedReference==1){
                RoundTrips +=1;
                TouchedReference=0;
            }


        }//sweep_no




        // double Temp_ = TemperatureReplicaSets[temp_set][Lowest_Temp_index];
        // double std_E = max(0.0,(((TotalE_square_Mean_ / (Confs_used * 1.0)) - ((TotalE_Mean_ * TotalE_Mean_) / (Confs_used * Confs_used * 1.0)))));
        // cout<<"Temperature, Energy, std(Energy) : "<<setw(15)<<Temp_<<setw(15)<<(TotalE_Mean_/Confs_used)<<setw(15)<<std_E<<endl;
        // cout<<"Temperature, Cv :"<<setw(15)<<Temp_<<setw(15)<<std_E/(Parameters_.Boltzman_constant*Parameters_.Boltzman_constant*Temp_*Temp_*ns_)<<endl;


        // for(int Ti=0;Ti<N_temperature_slices;Ti++){
        // char temp_char[50];
        // sprintf(temp_char, "%.10f", TemperatureReplicaSets[temp_set][Ti]);
        // string ObsOutFile_str="LocalObsOut_TemperatureSet"+ to_string(temp_set)+ "TemperatureValue"+ string(temp_char) +".txt";
        // ofstream ObsOutFile(ObsOutFile_str.c_str());
        // ObsOutFile<<"#site Sx Sy Sz Sx2 Sy2 Sz2"<<endl;
        // for(int site=0;site<ns_;site++){
        //     ObsOutFile<<site<<"  "<<GetLocalOprExp("Sx", site, Ti).real()<<"  "<<GetLocalOprExp("Sx", site, Ti).imag()<<"  "
        //                <<GetLocalOprExp("Sy", site, Ti).real()<<"  "<<GetLocalOprExp("Sy", site, Ti).imag()<<"  "
        //                <<GetLocalOprExp("Sz", site, Ti).real()<<"  "<<GetLocalOprExp("Sz", site, Ti).imag()<<"  "
        //                <<GetLocalOprExp("Sx2", site, Ti).real()<<"  "<<GetLocalOprExp("Sx2", site, Ti).imag()<<"  "
        //                <<GetLocalOprExp("Sy2", site, Ti).real()<<"  "<<GetLocalOprExp("Sy2", site, Ti).imag()<<"  "
        //                <<GetLocalOprExp("Sz2", site, Ti).real()<<"  "<<GetLocalOprExp("Sz2", site, Ti).imag()<<"  "
        //                <<endl;
        // }
        // }

        char temp_char[50];
        sprintf(temp_char, "%.10f", TemperatureReplicaSets[temp_set][Replica_Permutation[0]]);
        string File_Out_real_space_corr = "Real_space_corr" + to_string(temp_set)+ "TemperatureValue"+ string(temp_char) + ".txt";
        ofstream File_Out_Real_Space_Corr(File_Out_real_space_corr.c_str());

        File_Out_Real_Space_Corr<<"#i"<<setw(15)<<"j"<<setw(15)<<"<SS(i,j)>"<<setw(15)<<"sd(SS(i,j))"<<setw(15)<<"<TzTz(i,j)>"<<setw(15)<<"sd(TzTz(i,j))"<<endl;
        for (int site_i = 0; site_i < ns_; site_i++)
        {
            for (int site_j = 0; site_j < ns_; site_j++)
            {
                double std_SS = max(0.0,(((SiSj_square_Mean_(site_i, site_j)) - ((SiSj_Mean_(site_i, site_j) * SiSj_Mean_(site_i, site_j))) )));
                double std_TzTz = max(0.0,(((TauZiTauZj_square_Mean_(site_i, site_j)) - ((TauZiTauZj_Mean_(site_i, site_j) * TauZiTauZj_Mean_(site_i, site_j))))));
                File_Out_Real_Space_Corr << site_i << setw(15) << site_j << setw(15) << SiSj_Mean_(site_i, site_j)
                                         << setw(15) << sqrt(std_SS)
                                         << setw(15) << TauZiTauZj_Mean_(site_i, site_j)
                                         << setw(15) << sqrt(std_TzTz)
                                         <<endl;
            }
            File_Out_Real_Space_Corr << endl;
        }



        for(int ti=0;ti<N_temperature_slices-1;ti++){
            cout<<"No of swaps "<<ti<<" "<< No_of_swaps[ti]<<endl;
        }


        int Ti_temp = Replica_Permutation[0];
        char temp_char2[50];
        sprintf(temp_char2, "%.10f", TemperatureReplicaSets[temp_set][Ti_temp]);
        string ObsOutFile_str="MicrostateLocalObs_TemperatureSet"+ to_string(temp_set)+ "TemperatureValue"+ string(temp_char) +".txt";
        ofstream ObsOutFile(ObsOutFile_str.c_str());
        ObsOutFile<<"#site Sx Sy Sz Sx2 Sy2 Sz2  d1  d2  d3"<<endl;
        for(int site=0;site<ns_;site++){
            ObsOutFile<<site<<setw(15)<<GetLocalOprExp("Sx", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("Sx", site, Ti_temp).imag()<<setw(15)
            <<GetLocalOprExp("Sy", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("Sy", site, Ti_temp).imag()<<setw(15)
            <<GetLocalOprExp("Sz", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("Sz", site, Ti_temp).imag()<<setw(15)
            <<GetLocalOprExp("Sx2", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("Sx2", site, Ti_temp).imag()<<setw(15)
            <<GetLocalOprExp("Sy2", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("Sy2", site, Ti_temp).imag()<<setw(15)
            <<GetLocalOprExp("Sz2", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("Sz2", site, Ti_temp).imag();

            if(SiteTags[site]=="CP2"){
                ObsOutFile<<setw(15)
                <<GetLocalOprExp("d1", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("d1", site, Ti_temp).imag()<<setw(15)
                <<GetLocalOprExp("d2", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("d2", site, Ti_temp).imag()<<setw(15)
                <<GetLocalOprExp("d3", site, Ti_temp).real()<<setw(15)<<GetLocalOprExp("d3", site, Ti_temp).imag();
            }

            ObsOutFile<<endl;
        }






    } //Temperature Set








} // ---------



#endif // MCENGINE_H
