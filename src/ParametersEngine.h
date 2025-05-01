#ifndef Parameters_class
#define Parameters_class
#include "tensor_type.h"

class Parameters
{

public:
    int ns, IterMax, MCNorm, RandomSeed;
    string ModelType;
    double Boltzman_constant;


    int N_ConnectionsFiles;
    Mat_1_string ConnectionFiles;
    Mat_2_string Connections;


    Mat_1_doub Temp_values;
    bool Read_Seed_from_file_;
    string Seed_file_name_;

    bool Cooling_;

    bool Metropolis_Algorithm;
    bool Heat_Bath_Algorithm;

    bool PTMC; //Parallel Tempering Monte Carlo
    int N_replica_sets;
    int N_temperature_slices;
    int NProcessors;
    int SwappingSweepGap;
    Mat_2_doub TemperatureReplicaSets;

    /*
SavingMicroscopicStates=1
NoOfMicroscopicStates=50
      */
    bool Saving_Microscopic_States;
    int No_Of_Microscopic_States;

    double temp_max, beta_min;
    double temp_min, beta_max;
    double d_Temp;

    int Last_n_sweeps_for_measurement;
    int Measurement_after_each_m_sweeps;

    double temp, beta, Eav;
    double WindowSize, AccCount[2];
    char Dflag;

    string File_SiteTags;

    void Initialize(string inputfile_);
    double matchstring(string file, string match);
    string matchstring2(string file, string match);
};

void Parameters::Initialize(string inputfile_)
{

    double cooling_double;
    double Read_Seed_from_file_double;
    double metropolis_double;
    int SavingMicroscopicStates_int;
    int no_of_temp_points;
    string temp_values_;


    string genericconnectionsfiles_, GenericConnectionsFiles_ = "GenericConnectionsFiles = ";


    cout << "____________________________________" << endl;
    cout << "Reading the inputfile: " << inputfile_ << endl;
    cout << "____________________________________" << endl;


    NProcessors=int(matchstring(inputfile_,"NProcessors"));

    ns = int(matchstring(inputfile_, "Nsites"));
    SavingMicroscopicStates_int = int(matchstring(inputfile_, "SavingMicroscopicStates"));

    assert(SavingMicroscopicStates_int == 1 ||
           SavingMicroscopicStates_int == 0);
    if (SavingMicroscopicStates_int == 1)
    {
        Saving_Microscopic_States = true;
    }
    else
    {
        Saving_Microscopic_States = false;
    }

    No_Of_Microscopic_States = int(matchstring(inputfile_, "NoOfMicroscopicStates"));

    cout << "TotalNumberOfSites = " << ns << endl;

    IterMax = int(matchstring(inputfile_, "MaxMCsweeps"));
    MCNorm = 0.0; //matchstring(inputfile,"MCNorm")
    RandomSeed = matchstring(inputfile_, "RandomSeed");
    Boltzman_constant = matchstring(inputfile_, "Boltzman_constant");



    Dflag = 'N';

    metropolis_double = double(matchstring(inputfile_, "Metropolis_Algo"));
    if (metropolis_double == 1.0)
    {
        cout<<"Metropolis used"<<endl;
        Metropolis_Algorithm = true;
        Heat_Bath_Algorithm = false;
    }
    else if (metropolis_double == 0.0)
    {
        cout<<"Heat Bath used"<<endl;
        Metropolis_Algorithm = false;
        Heat_Bath_Algorithm = true;
    }

    else
    {
        cout << "ERROR: Metropolis_Algo can be only 1 (true) or 0 (false)" << endl;
        assert(metropolis_double == 0.0);
    }


    cooling_double = double(matchstring(inputfile_, "Cooling"));
    if (cooling_double == 1.0)
    {
        Cooling_ = true;

        temp_min = double(matchstring(inputfile_, "Temperature_min"));
        temp_max = double(matchstring(inputfile_, "Temperature_max"));
        d_Temp = double(matchstring(inputfile_, "dTemperature"));
        beta_max = double(Boltzman_constant / temp_min);
        beta_min = double(Boltzman_constant / temp_max);

        no_of_temp_points = int( (( temp_max - temp_min )/(d_Temp)) + 1);
        Temp_values.resize(no_of_temp_points);
        for(int point_no=0;point_no<no_of_temp_points;point_no++){
            Temp_values[point_no] = temp_max - (point_no*(d_Temp));
        }

    }
    else if (cooling_double == 2.0)
    {
        Cooling_ = true;
        temp_values_ = matchstring2(inputfile_, "Temperature_Values");

        stringstream temp_values_stream(temp_values_);
        temp_values_stream>>no_of_temp_points;

        Temp_values.resize(no_of_temp_points);

        for(int point_no=0;point_no<no_of_temp_points;point_no++){
            temp_values_stream >> Temp_values[point_no];
        }

    }
    else if (cooling_double == 3.0){
        PTMC = true;

        cout<<"XXXXXXXX Parallel Tempering is used XXXXXXXXXXXX"<<endl;

        string replica_info = matchstring2(inputfile_, "Replica_Info");
        stringstream replica_info_stream(replica_info);

        replica_info_stream>>N_replica_sets;
        replica_info_stream>>N_temperature_slices;
        replica_info_stream>>SwappingSweepGap;

        TemperatureReplicaSets.resize(N_replica_sets);

        string replica_temps = matchstring2(inputfile_, "Replica_Temperatures");
        stringstream replica_temps_stream(replica_temps);
        for(int Rset=0;Rset<N_replica_sets;Rset++){
            TemperatureReplicaSets[Rset].resize(N_temperature_slices);
            for(int Ti=0;Ti<N_temperature_slices;Ti++){
                replica_temps_stream>>TemperatureReplicaSets[Rset][Ti];
            }
        }
    }
    else if (cooling_double == 0.0)
    {
        Cooling_ = false;

        temp = double(matchstring(inputfile_, "Temperature")); // temperature in kelvin
        beta = double(1.0/(Boltzman_constant*temp));                         //Beta which is (T*k_b)^-1

        temp_min = temp;
        temp_max = temp;
        d_Temp = 10.0; //arbitrary positive number

        Temp_values.resize(1);
        Temp_values[0]=temp_min;
    }
    else
    {
        cout << "ERROR: Cooling can be only 1 or 2 (true) or 0 (false)" << endl;
        assert(cooling_double == 0.0);
    }


    Read_Seed_from_file_double = double(matchstring(inputfile_, "Read_Seed_from_file"));
    if (Read_Seed_from_file_double == 1.0)
    {
        Read_Seed_from_file_ = true;
    }
    else if (Read_Seed_from_file_double == 0.0)
    {
        Read_Seed_from_file_ = false;
    }
    else
    {
        cout << "ERROR:Read_Seed_from_file can be only 1 (true) or 0 (false)" << endl;
        assert(Read_Seed_from_file_double == 0.0);
    }


    Seed_file_name_ = matchstring2(inputfile_, "Seed_file_name");

    Last_n_sweeps_for_measurement = int(matchstring(inputfile_, "Last_n_sweeps_for_measurement"));
    Measurement_after_each_m_sweeps = int(matchstring(inputfile_, "Measurement_after_each_m_sweeps"));



    File_SiteTags = matchstring2(inputfile_, "File_SiteTags");

    genericconnectionsfiles_ = matchstring2(inputfile_, "GenericConnectionsFiles");
    stringstream genericconnectionsfiles_stream;
    genericconnectionsfiles_stream<<genericconnectionsfiles_;
    genericconnectionsfiles_stream>>N_ConnectionsFiles;
    ConnectionFiles.resize(N_ConnectionsFiles);
    string filename_temp;
    for(int i=0;i<N_ConnectionsFiles;i++){
        genericconnectionsfiles_stream>>filename_temp;
        ConnectionFiles[i]=filename_temp;
    }


    Connections.resize(N_ConnectionsFiles);
    for(int FileNo=0;FileNo<N_ConnectionsFiles;FileNo++){
        string line_connection;
        ifstream inputfileConnection(ConnectionFiles[FileNo].c_str());
        Connections[FileNo].clear();
        while(getline(inputfileConnection,line_connection)){
            Connections[FileNo].push_back(line_connection);
        }
    }



    //pi = 4.00 * atan(double(1.0));

    Eav = 0.0;
    AccCount[0] = 0;
    AccCount[1] = 0;

    WindowSize = double(0.01);
    cout << "____________________________________" << endl;
}

double Parameters::matchstring(string file, string match)
{
    string test;
    string line;
    ifstream readFile(file);
    double amount;
    bool pass = false;
    while (std::getline(readFile, line))
    {
        std::istringstream iss(line);
        if (std::getline(iss, test, '=') && pass == false)
        {
            // ---------------------------------
            if (iss >> amount && test == match)
            {
                // cout << amount << endl;
                pass = true;
            }
            else
            {
                pass = false;
            }
            // ---------------------------------
            if (pass)
                break;
        }
    }
    if (pass == false)
    {
        string errorout = match;
        errorout += "= argument is missing in the input file!";
        throw std::invalid_argument(errorout);
    }
    cout << match << " = " << amount << endl;
    return amount;
}

string Parameters::matchstring2(string file, string match)
{

    string line;
    ifstream readFile(file);
    string amount;
    int offset;

    if (readFile.is_open())
    {
        while (!readFile.eof())
        {
            getline(readFile, line);

            if ((offset = line.find(match, 0)) != string::npos)
            {
                amount = line.substr(offset + match.length() + 1);
            }
        }
        readFile.close();
    }
    else
    {
        cout << "Unable to open input file while in the Parameters class." << endl;
    }

    cout << match << " = " << amount << endl;
    return amount;
}

#endif
