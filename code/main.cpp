#include <iostream>
#include "DNest4/code/Start.h"
#include "MyModel.h"
#include "Data.h"

using namespace std;
using namespace DNest4;

int main(int argc, char** argv)
{
    // Process command line options
    CommandLineOptions clo(argc, argv);

    // Get specified data file. If none, use a default.
    std::string data_file = clo.get_data_file();
    if(data_file.length() == 0)
        data_file = std::string("../data/sample_data.txt");

    // Save the data filename
    std::fstream fout("run_data.txt", std::ios::out);
    fout<<data_file;
    fout.close();

    // Load data
    Data::get_instance().load(data_file.c_str());

    // Run DNest4.
    start<MyModel>(clo);

	return 0;
}

