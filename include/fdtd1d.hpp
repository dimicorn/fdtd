#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace phys {

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double EPS0 = 8.8541878128e-12;
inline constexpr double MU0 = 4.0e-7 * PI;

inline const double C0 = 1.0 / std::sqrt(EPS0 * MU0);

}

class Fdtd1D {
public:
	Fdtd1D(int N, double dx, double dt);

	void set_material_region(int i_start, int i_end, double eps_inf,
	                         double eps_s, double omega0, double delta);

	void set_source(int i_src, double amplitude, double t0, double tau,
	                double f0, double phase);

	void set_abc(double v_left, double v_right);

	void step();

	void write_snapshot_csv(const std::string& path,
	                        const std::string& scenario_name) const;

	double time() const { return static_cast<double>(step_count_) * dt_; }
	int num_cells() const { return N_; }
	double dx() const { return dx_; }
	double dt() const { return dt_; }
	const std::vector<double>& ez() const { return Ez_; }

private:
	double source_waveform(double t) const;

	int N_;
	double dx_, dt_;

	std::vector<double> Ez_, Hy_, Dz_, Pz_, Pz_prev_, Pz_scratch_;
	std::vector<double> eps_inf_, eps_s_, omega0_, delta_;

	int i_src_ = -1;
	double src_amp_ = 0.0, src_t0_ = 0.0, src_tau_ = 1.0;
	double src_f0_ = 0.0, src_phase_ = 0.0;

	double coeff_left_ = 0.0, coeff_right_ = 0.0;

	long step_count_ = 0;
};
