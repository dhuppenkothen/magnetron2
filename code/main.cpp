#include <iostream>
#include "DNest4/code/Start.h"
#include "MyModel.h"
#include "Data.h"

using namespace std;
using namespace DNest4;

int main(int argc, char** argv)
{
	Data::get_instance().load("../data/J1104+3812.txt");
	Sampler<MyModel> sampler = setup<MyModel>(argc, argv);
	sampler.run();
	return 0;
}

