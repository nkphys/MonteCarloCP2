#include <iostream>  //for cin and cout
#include <math.h>  // for pow
#include <stdlib.h>  //for div(q,n).rem(quot),abs(int n)
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>
#include "tensor_type.h"

int main(){




    string SpinTag="O3";
    string TauTag="O3";    

    double tx=0.6;
    double ty=0.6;
    double tz=1.0;

    double U=20.0;
    double JbyU=0.275;
    double J;
    J=U*JbyU;


    int N; 

  //8x8 PBCxPBC--------------------------

    int Lx=4;
   int Ly=4;
   N=Lx*Ly;
   
   //index = x +y*Lx
   Mat_1_int Bonds_Ytype_site1, Bonds_Ytype_site2;
   Mat_1_int Bonds_Xtype_site1, Bonds_Xtype_site2;

   int jx,jy,site_j,site_i;
   for(int ix=0;ix<Lx;ix++){
   for(int iy=0;iy<Ly;iy++){
   site_i=ix+iy*Lx;

   //+X neigh
   jx=(ix+1)%Lx;
   jy=iy;
   site_j = jx + jy*Lx;
   Bonds_Xtype_site1.push_back(site_i);Bonds_Xtype_site2.push_back(site_j);

   //+Y neigh
   jy=(iy+1)%Ly;
   jx=ix;
   site_j = jx + jy*Lx;
   Bonds_Ytype_site1.push_back(site_i);Bonds_Ytype_site2.push_back(site_j);

   }
   }
   
  //-----------------------------------------


    string H_SSConnfile_str = "H_SS.dat";
    ofstream H_SSConn_file_stream(H_SSConnfile_str.c_str());

    string H_SS_TTConnfile_str = "H_SS_TT.dat";
    ofstream H_SS_TTConn_file_stream(H_SS_TTConnfile_str.c_str());

    string H_SS_TConnfile_str = "H_SS_T.dat";
    ofstream H_SS_TConn_file_stream(H_SS_TConnfile_str.c_str());

    string H_TTConnfile_str = "H_TT.dat";
    ofstream H_TTConn_file_stream(H_TTConnfile_str.c_str());

    string SiteTagsfile_str = "SiteTags.dat";
    ofstream SiteTags_file_stream(SiteTagsfile_str.c_str());

    for(int i=0;i<N;i++){
	SiteTags_file_stream<<i<<"  "<<SpinTag<<"  "<<1.0<<endl;
    }
    for(int i=0;i<N;i++){
        SiteTags_file_stream<<i+N<<"  "<<TauTag<<"  "<<0.5<<endl;
    }




    int site_neigh;
    int site;

    Mat_1_string Neigh_type;
    Mat_1_int x_offset, y_offset;
    Mat_1_doub hop_x_or_y;
    Mat_1_string opr_str, opr_str2;
    Neigh_type.push_back("plusX");x_offset.push_back(1);y_offset.push_back(0);hop_x_or_y.push_back(tx);
    Neigh_type.push_back("plusY");x_offset.push_back(0);y_offset.push_back(1);hop_x_or_y.push_back(ty);
    opr_str.push_back("y");opr_str.push_back("x");
    opr_str2.push_back("x");opr_str2.push_back("y");


    string str_temp;

    double fac_dir;

    //H_SSTT + H_TT
    for(int neigh_type=0;neigh_type<Neigh_type.size();neigh_type++){
        double txy=hop_x_or_y[neigh_type];

        Mat_1_int Bonds_site1, Bonds_site2;
        if(Neigh_type[neigh_type]=="plusX"){
            Bonds_site1 = Bonds_Xtype_site1;
            Bonds_site2 = Bonds_Xtype_site2;
	    fac_dir=1.0;
        }
        if(Neigh_type[neigh_type]=="plusY"){
            Bonds_site1 = Bonds_Ytype_site1;
            Bonds_site2 = Bonds_Ytype_site2;
	    fac_dir=-1.0;
        }

        for(int bond_no=0;bond_no<Bonds_site1.size();bond_no++){
            site=Bonds_site1[bond_no];
            site_neigh=Bonds_site2[bond_no];


            Mat_1_string OPR_type;
            Mat_1_doub Hop_type;

            //str_temp = "S"+opr_str[neigh_type] + "2 " + "S"+opr_str[neigh_type] + "2";
            //OPR_type.push_back(str_temp);Hop_type.push_back(txy);
            OPR_type.push_back("Sz Sz");
            Mat_1_string SS_type;
            Mat_1_doub fac;
            SS_type.push_back("Sz Sz");fac.push_back(1.0);
            SS_type.push_back("Sp Sm");fac.push_back(0.5);
            SS_type.push_back("Sm Sp");fac.push_back(0.5);

            //TTSS 
              for(int type=0;type<3;type++){
                H_SS_TTConn_file_stream<<"4 "<<SS_type[type]<<" "<<"Sz Sz"<<" ";
                H_SS_TTConn_file_stream<<site<<" "<<site_neigh<<" "<<site+N<<" "<<site_neigh+N<<" ";
                H_SS_TTConn_file_stream<<fac[type]*txy*txy*(1.0/U)*( ((U+J)/(U+2.0*J)) + (2*J/(U-3.0*J)) )<<endl;
              }
                
            
	   //+/-(Ti+Tj)SS 
              for(int type=0;type<3;type++){
                H_SS_TConn_file_stream<<"3 "<<SS_type[type]<<" "<<"Sz"<<" ";
                H_SS_TConn_file_stream<<site<<" "<<site_neigh<<" "<<site+N<<" ";
                H_SS_TConn_file_stream<<fac_dir*fac[type]*txy*txy*(1.0/U)*( ((U+J)/(2.0*(U+2.0*J))) )<<endl;
	      }
	      for(int type=0;type<3;type++){
                H_SS_TConn_file_stream<<"3 "<<SS_type[type]<<" "<<"Sz"<<" ";
                H_SS_TConn_file_stream<<site<<" "<<site_neigh<<" "<<site_neigh+N<<" ";
                H_SS_TConn_file_stream<<fac_dir*fac[type]*txy*txy*(1.0/U)*( ((U+J)/(2.0*(U+2.0*J))) )<<endl;
              }


	      //SS
              for(int type=0;type<3;type++){
                H_SSConn_file_stream<<"2 "<<SS_type[type]<<" ";
                H_SSConn_file_stream<<site<<" "<<site_neigh<<" ";
                H_SSConn_file_stream<<(fac[type]*txy*txy*(1.0/U)*( ((U+J)/(4*(U+2.0*J))) - (J/(2*(U-3.0*J))) )  )  + 
		      		         (fac[type]*tz*tz*(1.0/U)*( (U+J)/(U+2.0*J) ) )	<<endl;
              }


	      //TT
	    H_TTConn_file_stream<<"2 "<<"Sz Sz"<<" ";
                H_TTConn_file_stream<<site+N<<" "<<site_neigh+N<<" ";
              H_TTConn_file_stream<<txy*txy*(1.0/U)*( ( (2*(U-J))/(U-3.0*J) )  - ((U+J)/(U+2.0*J)) )<<endl;


    
	}

    }//neigh



    return 0;
}
