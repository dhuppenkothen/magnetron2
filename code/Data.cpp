#include "Data.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

Data Data::instance;

Data::Data()
{

}

void Data::load(const char* filename)
{
	fstream fin(filename, ios::in);
	if(!fin)
	{
		cerr<<"# Failed to open file "<<filename<<"."<<endl;
		return;
	}

	t.clear();
	y.clear();
	yerr.clear();

	double temp1, temp2, temp3;
	while(fin>>temp1 && fin>>temp2 && fin>>temp3)
	{
		t.push_back(temp1);
		y.push_back(temp2);
		yerr.push_back(temp3);
	}

	fin.close();
	cout<<"# Found "<<t.size()<<" points in file "<<filename<<"."<<endl;

	compute_summaries();
}

void Data::compute_summaries()
{
	t_min = *min_element(t.begin(), t.end());
	t_max = *max_element(t.begin(), t.end());
	t_range = t_max - t_min;
	dt = t[1] - t[0];

	// Left and right edges of the data bins
//	t_left.assign(t.size(), 0.);
//	t_right.assign(t.size(), 0.);
//	for(size_t i=0; i<t.size(); i++)
//	{
//		t_left[i] = t[i] - 0.5*dt;
//		t_right[i] = t[i] + 0.5*dt;
//	}
}

