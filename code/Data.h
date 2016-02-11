#ifndef _Data_
#define _Data_

#include <vector>

class Data
{
	private:
		std::vector<double> t, y;
		std::vector<double> t_left, t_right;

		// Some useful summaries
		double t_min, t_max, t_range, dt;
		double y_mean;
		void compute_summaries();

	public:
		Data();
		void load(const char* filename);

		// Getters
		const std::vector<double>& get_t() const { return t; }
		const std::vector<double>& get_t_left() const { return t_left; }
		const std::vector<double>& get_t_right() const { return t_right; }
		const std::vector<double>& get_y() const { return y; }
		double get_t_min() const { return t_min; }
		double get_t_max() const { return t_max; }
		double get_t_range() const { return t_range; }
		double get_dt() const { return dt; }

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

