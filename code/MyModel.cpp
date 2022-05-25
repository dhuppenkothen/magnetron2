#include "MyModel.h"
#include "DNest4/code/Utils.h"
#include "Data.h"
#include <cmath>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_erf.h>

using namespace std;
using namespace DNest4;

const Data& MyModel::data = Data::get_instance();
#include <iostream>

MyModel::MyModel()
:bursts(4, 100, false, MyConditionalPrior(data.get_t_min(), data.get_t_max(),
                1E-10, 5.0*3.5e5*data.get_dt()))
//,noise_normals(data.get_t().size())
,mu(data.get_t().size())
{

}

void MyModel::calculate_mu()
{
        const vector<double>& t = data.get_t();

        // Update or from scratch?
        bool update = (bursts.get_added().size() < bursts.get_components().size());

        // Get the components
        const vector< vector<double> >& components = (update)?(bursts.get_added()):
                                (bursts.get_components());

        // Set the background level
        if(!update)
                mu.assign(mu.size(), background);

        double amplitude, skew, tc;
        double rise;
        double tpar;
        double cdf, pdf;
        double erf_inner, pdf_fac;

        for(size_t j=0; j<components.size(); j++)
        {
                tc = components[j][0];
                amplitude = components[j][1];
                rise = components[j][2];
                skew = components[j][3];


                for(size_t i=0; i<mu.size(); i++)
                {
                        
                        tpar = (t[i] - tc); 
                        erf_inner = skew * (tpar/rise) / sqrt(2);
                        cdf = 0.5 * (1.0 + gsl_sf_erf(erf_inner));
                        pdf_fac = 1.0 / (rise * pow(2*M_PI, 0.5));
                        pdf = pdf_fac * exp(-pow(tpar, 2) / (2*pow(rise, 2)));
                        mu[i] += amplitude * 2 * pdf * cdf;

                }

        }

}

void MyModel::from_prior(RNG& rng)
{
	background = tan(M_PI*(0.97*rng.rand() - 0.485));
	background = exp(background);
	bursts.from_prior(rng);
 
//        noise_sigma = exp(log(1E-3) + log(1E3)*rng.rand());
//        noise_L = exp(log(1E-2*Data::get_instance().get_t_range())
//                        + log(1E3)*rng.rand());

        calculate_mu();

}

double MyModel::perturb(RNG& rng)
{
	double logH = 0.;

        if(rng.rand() <= 0.2)
        {
                for(size_t i=0; i<mu.size(); i++)
                        mu[i] -= background;

                background = log(background);
                background = (atan(background)/M_PI + 0.485)/0.97;
                background += pow(10., 1.5 - 6.*rng.rand())*rng.randn();
                background = mod(background, 1.);
                background = tan(M_PI*(0.97*background - 0.485));
                background = exp(background);

                for(size_t i=0; i<mu.size(); i++)
                        mu[i] += background;
        }

//        else if(rng.rand() <= 0.5)
//        {
//                noise_sigma = log(noise_sigma);
//                noise_sigma += log(1E3)*rng.randh();
//                wrap(noise_sigma, log(1E-3), log(1.));
//                noise_sigma = exp(noise_sigma);
//
//                noise_L = log(noise_L);
//                noise_L += log(1E3)*rng.randh();
//                wrap(noise_L, log(1E-2*Data::get_instance().get_t_range()), log(10.*Data::get_instance().get_t_range()));
//                noise_L = exp(noise_L);
//
//                calculate_mu();
//        }
        else
        {
//                int num = exp(log((double)noise_normals.size())*rng.rand());
//                for(int i=0; i<num; i++)
//                {
//                        int k = rng.rand_int(noise_normals.size());
//                        noise_normals[k] = rng.randn();
//                }
                logH += bursts.perturb(rng);  
                bursts.consolidate_diff();

                calculate_mu();
        }


	return logH;
}

double MyModel::log_likelihood() const
{
        const vector<double>& t = data.get_t();
        const vector<double>& y = data.get_y();
        const vector<double>& yerr = data.get_yerr();

        double logl = 0.;
        for(size_t i=0; i<t.size(); i++)
//                logl += -mu[i] + y[i]*log(mu[i]) - lgamma(y[i] + 1.);//gsl_sf_lngamma(y[i] + 1.);
                  logl += -0.5 * log(2.*M_PI) - log(yerr[i]) - 0.5 * pow((y[i] - mu[i]) / yerr[i], 2);
	return logl;
}

void MyModel::print(std::ostream& out) const
{
        out<<background<<' ';
        bursts.print(out);
        for(size_t i=0; i<mu.size(); i++)
                out<<mu[i]<<' ';

}

string MyModel::description() const
{
	return string("");
}

