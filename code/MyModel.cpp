#include "MyModel.h"
#include "DNest4/code/Utils.h"
#include "Data.h"
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace DNest4;

const Data& MyModel::data = Data::get_instance();
#include <iostream>

// Initialise the static distribution
const DNest4::Cauchy MyModel::cauchy(0.0, 1.0);

MyModel::MyModel()
:bursts(4, 100, false, MyConditionalPrior(data.get_t_min(), data.get_t_max(),
                1E-10, 150.0 ))
,noise_normals(data.get_t().size())
,mu(data.get_t().size())
{

}

void MyModel::calculate_mu()
{
//        const vector<double>& t_left = data.get_t_left();
//        const vector<double>& t_right = data.get_t_right();
	const vector<double>& t = data.get_t();

        // Get the components
        const vector< vector<double> >& components = bursts.get_components();


        mu.assign(mu.size(), background);

        double amplitude, skew, tc;
        double rise, fall;

        for(size_t j=0; j<components.size(); j++)
        {
                tc = components[j][0];
                amplitude = components[j][1];
                rise = components[j][2];
                skew = components[j][3];

                fall = rise*skew;

                for(size_t i=0; i<mu.size(); i++)
                {
                        if(tc <= t[i])
                        {
                                // Bin to the right of peak
//                                mu[i] += amplitude*fall/data.get_dt()*
//                                                (exp((tc - t_left[i])/fall) -
//                                                 exp((tc - t_right[i])/fall));
				mu[i] += amplitude*exp((tc - t[i])/fall);
 
                        }
                        else
                        {
                                // Bin to the left of peak
//                                mu[i] += -amplitude*rise/data.get_dt()*
//                                                (exp((t_left[i] - tc)/rise) -
//                                                 exp((t_right[i] - tc)/rise));
				mu[i] += amplitude*exp((t[i] - tc)/rise);
                        }
                }
        vector<double> y(mu.size());
        double alpha = exp(-1./noise_L);

        for(size_t i=0; i<mu.size(); i++)
        {
                if(i==0)
                        y[i] = noise_sigma/sqrt(1. - alpha*alpha)*noise_normals[i];
                else
                        y[i] = alpha*y[i-1] + noise_sigma*noise_normals[i];
                mu[i] *= exp(y[i]);
        }

        }

}

void MyModel::from_prior(RNG& rng)
{
	background = tan(M_PI*(0.97*rng.rand() - 0.485));
	background = exp(background);
	bursts.from_prior(rng);
 
        noise_sigma = exp(log(1E-3) + log(1E3)*rng.rand());
        noise_L = exp(log(1E-2*Data::get_instance().get_t_range())
                        + log(1E3)*rng.rand());

        calculate_mu();

}

double MyModel::perturb(RNG& rng)
{
	double logH = 0.;

        int which;

        // A 50-50 decision to perturb a 'multivariate' thing or a 'univariate' thing
        if(rng.rand() <= 0.5)
        {
                which = rng.rand_int(2);

                if(which == 0)  // Proposal for the RJObject describing the flares
                        logH += bursts.perturb(rng);
                else                    // Proposal for normals for the OU process
                {
                        if(rng.rand() <= 0.5)   // Propose to move only one
                        {
                                int i = rng.rand_int(noise_normals.size());
                                logH -= -0.5*pow(noise_normals[i], 2);
                                noise_normals[i] += rng.randh();
                                logH += -0.5*pow(noise_normals[i], 2);
                        }
                        else                                    // Regenerate many
                        {
                int reps = (int)pow(noise_normals.size(), rng.rand());
                for(int i=0; i<reps; ++i)
                {
                    int k = rng.rand_int(noise_normals.size());
                    noise_normals[k] = rng.randn();
                }
		}
                }
        }
        else
        {
                which = rng.rand_int(3);
                if(which == 0)
                {
            background = log(background);
            logH += cauchy.perturb(background, rng);
            if(std::abs(background) > 25.0)
                logH = -1E300;
            background = exp(background);
                }
                else if(which == 1)
                {
                        noise_sigma = log(noise_sigma);
                        noise_sigma += log(1E3)*rng.randh();
                        wrap(noise_sigma, log(1E-3), log(1.));
                        noise_sigma = exp(noise_sigma);
                }
                else
                {
                        noise_L = log(noise_L);
                        noise_L += log(1E5)*rng.randh();
                        wrap(noise_L, log(0.01*Data::get_instance().get_t_range()), log(1000.*Data::get_instance().get_t_range()));
                        noise_L = exp(noise_L);
                }
        }

        calculate_mu();
        return logH;

//        if(rng.rand() <= 0.2)
//        {
//                for(size_t i=0; i<mu.size(); i++)
//                        mu[i] -= background;
//
//                background = log(background);
//                background = (atan(background)/M_PI + 0.485)/0.97;
//                background += pow(10., 1.5 - 6.*rng.rand())*rng.randn();
//                background = mod(background, 1.);
//                background = tan(M_PI*(0.97*background - 0.485));
//                background = exp(background);
//
//                for(size_t i=0; i<mu.size(); i++)
//                        mu[i] += background;
//        }
//
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
//        else
//        {
//                int num = exp(log((double)noise_normals.size())*rng.rand());
//                for(int i=0; i<num; i++)
//                {
//                        int k = rng.rand_int(noise_normals.size());
//                        noise_normals[k] = rng.randn();
//                }
//                calculate_mu();
//        }
//

//	return logH;
}

double MyModel::log_likelihood() const
{
        const vector<double>& t = data.get_t();
        const vector<double>& y = data.get_y();
	const vector<double>& yerr = data.get_yerr();

        double logl = 0.;
        for(size_t i=0; i<t.size(); i++)
//                logl += -mu[i] + y[i]*log(mu[i]) - lgamma(y[i] + 1.);//gsl_sf_lngamma(y[i] + 1.);
		logl += -0.5 * log(2.*M_PI) - log(yerr[i]) - 0.5 * pow((y[i] - mu[i])/yerr[i], 2);
	return logl;
}

void MyModel::print(std::ostream& out) const
{
        out<<background<<' '<<noise_sigma<<' '<<noise_L<<' ';
        bursts.print(out);
        for(size_t i=0; i<mu.size(); i++)
                out<<mu[i]<<' ';

}

string MyModel::description() const
{
	return string("");
}

