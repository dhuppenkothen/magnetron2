#include "MyConditionalPrior.h"
#include "DNest4/code/Utils.h"
#include "Data.h"
#include <cmath>

using namespace DNest4;

MyConditionalPrior::MyConditionalPrior(double x_min, double x_max, 
					double mu_min, double mu_max)
:x_min(x_min)
,x_max(x_max)
,mu_min(mu_min)
,mu_max(mu_max)
,min_width(0.1) // NOT SURE THIS IS CORRECT MIN_WIDTH!
{

}

void MyConditionalPrior::from_prior(RNG& rng)
{
	mu = tan(M_PI*(0.97*rng.rand() - 0.485));
	mu = exp(mu);
	mu_widths = exp(log(1E-3*(x_max - x_min)) + log(1E3)*rng.rand());

	a = -20. + 40.*rng.rand();
	b = 10.*rng.rand();
}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.;

	int which = rng.rand_int(4);

	if(which == 0)
	{
		mu = log(mu);
		mu = (atan(mu)/M_PI + 0.485)/0.97;
		mu += pow(10., 1.5 - 6.*rng.rand())*rng.randn();
		mu = mod(mu, 1.);
		mu = tan(M_PI*(0.97*mu - 0.485));
		mu = exp(mu);
	}
	if(which == 1)
	{
		mu_widths = log(mu_widths/(x_max - x_min));
		mu_widths += log(1E3)*pow(10., 1.5 - 6.*rng.rand())*rng.randn();
		mu_widths = mod(mu_widths - log(1E-3), log(1E3)) + log(1E-3);
		mu_widths = (x_max - x_min)*exp(mu_widths);
	}
	if(which == 2)
	{
		a += 40.*rng.randh();
		a = mod(a + 20., 40.) - 20.;
	}
	if(which == 3)
	{
		b += 10.*rng.randh();
		b = mod(b, 10.);
	}

	return logH;
}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
	if(vec[0] < x_min || vec[0] > x_max || vec[1] < 0.0 || vec[2] < min_width
		|| log(vec[3]) < (a-b) || log(vec[3]) > (a + b))
		return -1E300;

	return -log(mu) - vec[1]/mu - log(mu_widths)
			- (vec[2] - min_width)/mu_widths - log(2.*b*vec[3]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
	vec[0] = x_min + (x_max - x_min)*vec[0];
	vec[1] = -mu*log(1. - vec[1]);
	vec[2] = min_width - mu_widths*log(1. - vec[2]);
	vec[3] = exp(a - b + 2.*b*vec[3]);
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
	vec[0] = (vec[0] - x_min)/(x_max - x_min);
	vec[1] = 1. - exp(-vec[1]/mu);
	vec[2] = 1. - exp(-(vec[2] - min_width)/mu_widths);
	vec[3] = (log(vec[3]) + b - a)/(2.*b);
}

void MyConditionalPrior::print(std::ostream& out) const
{
	out<<mu<<' '<<mu_widths<<' '<<a<<' '<<b<<' ';
}

