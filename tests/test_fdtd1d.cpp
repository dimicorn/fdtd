#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "fdtd1d.hpp"

namespace fs = std::filesystem;

namespace {

constexpr double FS = 1.0e-15;
constexpr double UM = 1.0e-6;
constexpr double DX = 2.0e-8;

double courant_dt() { return 0.5 * DX / phys::C0; }

int idx(double x) { return static_cast<int>(std::llround(x / DX)); }

void run_until(Fdtd1D& g, double t_s) {
	while (g.time() < t_s - g.dt() / 2.0) g.step();
}

double energy(const Fdtd1D& g, double x_lo, double x_hi) {
	double s = 0.0;
	const auto& ez = g.ez();
	for (int i = 0; i <= g.num_cells(); ++i) {
		const double x = i * g.dx();
		if (x >= x_lo && x <= x_hi) s += ez[i] * ez[i];
	}
	return s * g.dx();
}

double centroid(const Fdtd1D& g, double x_lo, double x_hi) {
	double num = 0.0, den = 0.0;
	const auto& ez = g.ez();
	for (int i = 0; i <= g.num_cells(); ++i) {
		const double x = i * g.dx();
		if (x >= x_lo && x <= x_hi) {
			num += x * ez[i] * ez[i];
			den += ez[i] * ez[i];
		}
	}
	return num / den;
}

double max_abs(const Fdtd1D& g, double x_lo, double x_hi) {
	double m = 0.0;
	const auto& ez = g.ez();
	for (int i = 0; i <= g.num_cells(); ++i) {
		const double x = i * g.dx();
		if (x >= x_lo && x <= x_hi) m = std::max(m, std::abs(ez[i]));
	}
	return m;
}

bool all_finite(const Fdtd1D& g) {
	for (double e : g.ez()) {
		if (!std::isfinite(e)) return false;
	}
	return true;
}

Fdtd1D make_vacuum(double L, double x_src) {
	const int N = idx(L);
	Fdtd1D g(N, DX, courant_dt());
	g.set_material_region(0, N, 1.0, 1.0, 1.0, 0.0);
	g.set_source(idx(x_src), 1.0, 25.0 * FS, 5.0 * FS, 0.0, 0.0);
	g.set_abc(phys::C0, phys::C0);
	return g;
}

constexpr double kEpsInf = 2.25, kEpsS = 5.25, kOmega0 = 4.0e14;
constexpr double kF0 = 1.37e14;
constexpr double kXd = 80.0 * UM, kLorentzL = 170.0 * UM;

std::complex<double> lorentz_n(double w, double delta) {
	const std::complex<double> eps =
			kEpsInf +
			kOmega0 * kOmega0 * (kEpsS - kEpsInf) /
					std::complex<double>(kOmega0 * kOmega0 - w * w, -delta * w);
	return std::sqrt(eps);
}

double lorentz_group_index(double w, double delta) {
	const double h = 1e-4 * w;
	return lorentz_n(w, delta).real() +
	       w *
	               (lorentz_n(w + h, delta).real() -
	                lorentz_n(w - h, delta).real()) /
	               (2.0 * h);
}

Fdtd1D make_lorentz_half_space(double delta) {
	const int N = idx(kLorentzL);
	Fdtd1D g(N, DX, courant_dt());
	g.set_material_region(0, idx(kXd) - 1, 1.0, 1.0, 1.0, 0.0);
	g.set_material_region(idx(kXd), N, kEpsInf, kEpsS, kOmega0, delta);
	g.set_source(idx(40.0 * UM), 1.0, 240.0 * FS, 60.0 * FS, kF0, 0.0);
	g.set_abc(phys::C0, phys::C0 / std::sqrt(kEpsInf));
	return g;
}

}

TEST(Basics, ZeroFieldStaysZero) {
	Fdtd1D g(500, DX, courant_dt());
	g.set_material_region(250, 500, 2.25, 5.25, 4.0e14, 2.0e9);
	g.set_abc(phys::C0, phys::C0 / 1.5);
	for (int n = 0; n < 2000; ++n) g.step();
	EXPECT_EQ(max_abs(g, 0.0, 1.0), 0.0);
}

TEST(Basics, TimeBookkeeping) {
	Fdtd1D g(100, DX, courant_dt());
	EXPECT_EQ(g.time(), 0.0);
	for (int n = 0; n < 37; ++n) g.step();
	EXPECT_DOUBLE_EQ(g.time(), 37 * courant_dt());
	EXPECT_EQ(g.num_cells(), 100);
	EXPECT_EQ(g.ez().size(), 101u);
}

TEST(Basics, RejectsInvalidParameters) {
	EXPECT_THROW(Fdtd1D(1, DX, courant_dt()), std::invalid_argument);
	EXPECT_THROW(Fdtd1D(-5, DX, courant_dt()), std::invalid_argument);
	EXPECT_THROW(Fdtd1D(100, -DX, courant_dt()), std::invalid_argument);
	EXPECT_THROW(Fdtd1D(100, DX, 0.0), std::invalid_argument);
	EXPECT_THROW(Fdtd1D(100, DX, 1.01 * DX / phys::C0), std::invalid_argument);

	Fdtd1D g(100, DX, courant_dt());
	EXPECT_THROW(g.set_material_region(-1, 50, 1.0, 1.0, 1.0, 0.0),
	             std::out_of_range);
	EXPECT_THROW(g.set_material_region(50, 101, 1.0, 1.0, 1.0, 0.0),
	             std::out_of_range);
	EXPECT_THROW(g.set_material_region(0, 50, 0.0, 0.0, 1.0, 0.0),
	             std::invalid_argument);
	EXPECT_THROW(g.set_material_region(0, 50, 0.1, 0.1, 1.0, 0.0),
	             std::invalid_argument);
	EXPECT_THROW(g.set_source(0, 1.0, 0.0, 1.0, 0.0, 0.0), std::out_of_range);
	EXPECT_THROW(g.set_source(100, 1.0, 0.0, 1.0, 0.0, 0.0), std::out_of_range);
}

TEST(Vacuum, FreeSpaceSpeed) {
	Fdtd1D g = make_vacuum(30.0 * UM, 4.0 * UM);
	run_until(g, 45.0 * FS);
	const double x1 = centroid(g, 4.0 * UM, 30.0 * UM);
	const double t1 = g.time();
	run_until(g, 85.0 * FS);
	const double x2 = centroid(g, 4.0 * UM, 30.0 * UM);
	const double v = (x2 - x1) / (g.time() - t1);
	EXPECT_NEAR(v, phys::C0, 1e-3 * phys::C0);
}

TEST(Vacuum, LeftRightSymmetry) {
	Fdtd1D g = make_vacuum(30.0 * UM, 15.0 * UM);
	run_until(g, 60.0 * FS);
	const auto& ez = g.ez();
	const int c = idx(15.0 * UM);
	double max_diff = 0.0, max_e = 0.0;
	for (int k = 1; k < 700; ++k) {
		max_diff = std::max(max_diff, std::abs(ez[c + k] - ez[c - k]));
		max_e = std::max(max_e, std::abs(ez[c + k]));
	}
	ASSERT_GT(max_e, 0.0);
	EXPECT_LE(max_diff, 1e-9 * max_e);
}

TEST(Vacuum, MurAbcAbsorbs) {
	Fdtd1D g = make_vacuum(10.0 * UM, 5.0 * UM);
	run_until(g, 40.0 * FS);
	const double e_before = energy(g, 0.0, 10.0 * UM);
	run_until(g, 150.0 * FS);
	EXPECT_LT(energy(g, 0.0, 10.0 * UM), 1e-6 * e_before);
}

TEST(Vacuum, SoftSourceRadiatesGaussian) {
	Fdtd1D g = make_vacuum(30.0 * UM, 4.0 * UM);
	run_until(g, 45.0 * FS);
	double s = 0.0, s_abs = 0.0;
	const auto& ez = g.ez();
	for (int i = idx(4.0 * UM); i <= g.num_cells(); ++i) {
		s += ez[i];
		s_abs += std::abs(ez[i]);
	}
	EXPECT_NEAR(max_abs(g, 4.0 * UM, 30.0 * UM), 1.0, 0.05);
	EXPECT_GT(std::abs(s), 0.99 * s_abs);
}

TEST(Dielectric, FresnelEnergy) {
	const double eps = 2.25, n = std::sqrt(eps);
	const double L = 34.0 * UM, x_d = 14.0 * UM;
	const int N = idx(L);
	Fdtd1D g(N, DX, courant_dt());
	g.set_material_region(0, idx(x_d) - 1, 1.0, 1.0, 1.0, 0.0);
	g.set_material_region(idx(x_d), N, eps, eps, 1.0, 0.0);
	g.set_source(idx(4.0 * UM), 1.0, 25.0 * FS, 5.0 * FS, 0.0, 0.0);
	g.set_abc(phys::C0, phys::C0 / n);

	run_until(g, 38.36 * FS);
	const double e_inc = energy(g, 4.0 * UM, x_d);
	run_until(g, 93.36 * FS);
	const double R = energy(g, 0.0, x_d) / e_inc;
	const double Tr = energy(g, x_d, L) / e_inc;

	const double r = (1.0 - n) / (1.0 + n), t = 2.0 / (1.0 + n);
	EXPECT_NEAR(R, r * r, 0.005 * r * r);
	EXPECT_NEAR(Tr, t * t / n, 0.005 * t * t / n);
	EXPECT_NEAR(R + eps * Tr, 1.0, 1e-3);
}

TEST(Lorentz, StaticEqualsDielectric) {
	auto build = [](double omega0, double delta) {
		const int N = idx(34.0 * UM);
		Fdtd1D g(N, DX, courant_dt());
		g.set_material_region(0, idx(14.0 * UM) - 1, 1.0, 1.0, 1.0, 0.0);
		g.set_material_region(idx(14.0 * UM), N, 2.25, 2.25, omega0, delta);
		g.set_source(idx(4.0 * UM), 1.0, 25.0 * FS, 5.0 * FS, 0.0, 0.0);
		g.set_abc(phys::C0, phys::C0 / 1.5);
		run_until(g, 93.36 * FS);
		return g.ez();
	};
	EXPECT_EQ(build(1.0, 0.0), build(4.0e14, 2.0e9));
}

TEST(Lorentz, GroupVelocity) {
	const double delta = 2.0e9;
	Fdtd1D g = make_lorentz_half_space(delta);
	run_until(g, 560.0 * FS);
	const double x1 = centroid(g, kXd, kLorentzL);
	const double t1 = g.time();
	run_until(g, 720.0 * FS);
	const double x2 = centroid(g, kXd, kLorentzL);
	const double v = (x2 - x1) / (g.time() - t1);

	const double v_g =
			phys::C0 / lorentz_group_index(2.0 * phys::PI * kF0, delta);
	ASSERT_TRUE(all_finite(g));
	EXPECT_NEAR(v, v_g, 0.005 * v_g);
}

TEST(Lorentz, Damping) {
	const double delta = 3.0e13;
	const double w = 2.0 * phys::PI * kF0;
	Fdtd1D g = make_lorentz_half_space(delta);
	run_until(g, 560.0 * FS);
	const double e1 = energy(g, kXd, kLorentzL);
	const double t1 = g.time();
	run_until(g, 640.0 * FS);
	const double e2 = energy(g, kXd, kLorentzL);
	const double rate = -std::log(e2 / e1) / (g.time() - t1);

	const double alpha = w * lorentz_n(w, delta).imag() / phys::C0;
	const double v_g = phys::C0 / lorentz_group_index(w, delta);
	const double expected = 2.0 * alpha * v_g;
	EXPECT_NEAR(rate, expected, 0.03 * expected);
}

TEST(Lorentz, LongRunStable) {
	const int N = 1000;
	Fdtd1D g(N, DX, courant_dt());
	g.set_material_region(0, N, kEpsInf, kEpsS, kOmega0, 2.0e9);
	g.set_source(N / 2, 1.0, 75.0 * FS, 15.0 * FS, kF0, 0.0);
	g.set_abc(phys::C0 / 1.5, phys::C0 / 1.5);
	run_until(g, 150.0 * FS);
	const double m_early = max_abs(g, 0.0, 1.0);
	for (int n = 0; n < 200000; ++n) g.step();
	ASSERT_TRUE(all_finite(g));
	EXPECT_LE(max_abs(g, 0.0, 1.0), m_early);
}

TEST(Io, SnapshotCsvFormat) {
	Fdtd1D g = make_vacuum(1.0 * UM, 0.5 * UM);
	for (int n = 0; n < 10; ++n) g.step();
	const fs::path dir = fs::temp_directory_path() / "fdtd1d_test_snapshot";
	fs::create_directories(dir);
	const fs::path path = dir / "snap.csv";
	g.write_snapshot_csv(path.string(), "unit");

	std::ifstream in(path);
	std::string comment, header, line;
	std::getline(in, comment);
	std::getline(in, header);
	EXPECT_EQ(comment.rfind("# scenario=unit t_s=", 0), 0u);
	EXPECT_EQ(header, "x_m,Ez");

	int rows = 0;
	double last_x = -1.0;
	while (std::getline(in, line)) {
		double x = 0.0, e = 0.0;
		char comma = 0;
		std::istringstream ss(line);
		ss >> x >> comma >> e;
		ASSERT_FALSE(ss.fail()) << line;
		ASSERT_EQ(comma, ',');
		ASSERT_LE(rows, g.num_cells());
		EXPECT_NEAR(e, g.ez()[rows], 1e-8 * std::abs(g.ez()[rows]));
		last_x = x;
		++rows;
	}
	EXPECT_EQ(rows, g.num_cells() + 1);
	EXPECT_NEAR(last_x, g.num_cells() * DX, 1e-12 * g.num_cells() * DX);
	fs::remove_all(dir);
}
