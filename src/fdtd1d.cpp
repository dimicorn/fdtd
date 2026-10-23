#include "fdtd1d.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {

int checked_size(int N) {
	if (N < 2) throw std::invalid_argument("Fdtd1D: N must be >= 2");
	return N;
}

}

Fdtd1D::Fdtd1D(int N, double dx, double dt)
		: N_(checked_size(N)),
		  dx_(dx),
		  dt_(dt),
		  Ez_(N + 1, 0.0),
		  Hy_(N, 0.0),
		  Dz_(N + 1, 0.0),
		  Pz_(N + 1, 0.0),
		  Pz_prev_(N + 1, 0.0),
		  Pz_scratch_(N + 1, 0.0),
		  eps_inf_(N + 1, 1.0),
		  eps_s_(N + 1, 1.0),
		  omega0_(N + 1, 1.0),
		  delta_(N + 1, 0.0) {
	if (!(dx > 0.0) || !(dt > 0.0) || !std::isfinite(dx) ||
	    !std::isfinite(dt)) {
		throw std::invalid_argument("Fdtd1D: dx and dt must be positive");
	}
	if (phys::C0 * dt / dx > 1.0) {
		throw std::invalid_argument("Fdtd1D: Courant number c*dt/dx exceeds 1");
	}
}

void Fdtd1D::set_material_region(int i_start, int i_end, double eps_inf,
                                 double eps_s, double omega0, double delta) {
	if (i_start < 0 || i_end > N_ || i_start > i_end) {
		throw std::out_of_range("Fdtd1D: material region outside the grid");
	}
	if (!(eps_inf > 0.0)) {
		throw std::invalid_argument("Fdtd1D: eps_inf must be positive");
	}
	if (phys::C0 * dt_ / (dx_ * std::sqrt(eps_inf)) > 1.0) {
		throw std::invalid_argument(
				"Fdtd1D: Courant number exceeds 1 in a medium with eps_inf < "
				"1");
	}
	for (int i = i_start; i <= i_end; ++i) {
		eps_inf_[i] = eps_inf;
		eps_s_[i] = eps_s;
		omega0_[i] = omega0;
		delta_[i] = delta;
	}
}

void Fdtd1D::set_source(int i_src, double amplitude, double t0, double tau,
                        double f0, double phase) {
	if (i_src < 1 || i_src > N_ - 1) {
		throw std::out_of_range("Fdtd1D: source must be an interior node");
	}
	i_src_ = i_src;
	src_amp_ = amplitude;
	src_t0_ = t0;
	src_tau_ = tau;
	src_f0_ = f0;
	src_phase_ = phase;
}

void Fdtd1D::set_abc(double v_left, double v_right) {
	coeff_left_ = (v_left * dt_ - dx_) / (v_left * dt_ + dx_);
	coeff_right_ = (v_right * dt_ - dx_) / (v_right * dt_ + dx_);
}

double Fdtd1D::source_waveform(double t) const {
	const double arg = (t - src_t0_) / src_tau_;
	const double envelope = std::exp(-arg * arg);
	return src_amp_ * envelope *
	       std::cos(2.0 * phys::PI * src_f0_ * (t - src_t0_) + src_phase_);
}

void Fdtd1D::step() {
	const double ez0_old = Ez_[0];
	const double ez1_old = Ez_[1];
	const double ezNm1_old = Ez_[N_ - 1];
	const double ezN_old = Ez_[N_];

	for (int i = 0; i < N_; ++i) {
		Hy_[i] += (dt_ / (phys::MU0 * dx_)) * (Ez_[i + 1] - Ez_[i]);
	}

	for (int i = 1; i < N_; ++i) {
		Dz_[i] += (dt_ / dx_) * (Hy_[i] - Hy_[i - 1]);
	}

	for (int i = 1; i < N_; ++i) {
		const double a = 1.0 / (dt_ * dt_) + delta_[i] / (2.0 * dt_);
		const double b = 2.0 / (dt_ * dt_) - omega0_[i] * omega0_[i];
		const double c = 1.0 / (dt_ * dt_) - delta_[i] / (2.0 * dt_);
		const double drive = phys::EPS0 * omega0_[i] * omega0_[i] *
		                     (eps_s_[i] - eps_inf_[i]) * Ez_[i];
		Pz_scratch_[i] = (drive + b * Pz_[i] - c * Pz_prev_[i]) / a;
	}
	std::swap(Pz_prev_, Pz_);
	std::swap(Pz_, Pz_scratch_);

	for (int i = 1; i < N_; ++i) {
		Ez_[i] = (Dz_[i] - Pz_[i]) / (phys::EPS0 * eps_inf_[i]);
	}

	Ez_[0] = ez1_old + coeff_left_ * (Ez_[1] - ez0_old);
	Ez_[N_] = ezNm1_old + coeff_right_ * (Ez_[N_ - 1] - ezN_old);

	++step_count_;
	if (i_src_ >= 0) {
		const double s = source_waveform(time());
		Dz_[i_src_] += phys::EPS0 * eps_inf_[i_src_] * s;
		Ez_[i_src_] += s;
	}
}

void Fdtd1D::write_snapshot_csv(const std::string& path,
                                const std::string& scenario_name) const {
	std::ofstream out(path);
	if (!out) {
		throw std::runtime_error("could not open snapshot file: " + path);
	}
	out << std::scientific << std::setprecision(9);
	out << "# scenario=" << scenario_name << " t_s=" << time()
	    << " dx_m=" << dx_ << "\n";
	out << "x_m,Ez\n";
	for (int i = 0; i <= N_; ++i) {
		out << (static_cast<double>(i) * dx_) << "," << Ez_[i] << "\n";
	}
	out.close();
	if (!out) {
		throw std::runtime_error("could not write snapshot file: " + path);
	}
}
