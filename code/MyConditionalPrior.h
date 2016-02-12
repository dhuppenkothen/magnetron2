#ifndef _MyConditionalPrior_
#define _MyConditionalPrior_

#include "DNest4/code/RJObject/ConditionalPriors/ConditionalPrior.h"

class MyConditionalPrior:public DNest4::ConditionalPrior
{
	private:
		// Limits
		double x_min, x_max;
		double mu_min, mu_max;
		double min_width;

		// Mean of amplitudes and widths
		double mu, mu_widths;

		// Uniform for log-skews
		double a, b; // Midpoint and half-width

		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		MyConditionalPrior(double x_min, double x_max,
					double mu_min, double mu_max);

		void from_prior(DNest4::RNG& rng);

		double log_pdf(const std::vector<double>& vec) const;
		void from_uniform(std::vector<double>& vec) const;
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;

		static const int weight_parameter = 1;
};

#endif

