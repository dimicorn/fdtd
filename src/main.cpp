#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "fdtd1d.hpp"

namespace fs = std::filesystem;

namespace {

struct SnapshotSpec {
	double t_s;
	std::string label;
};

struct ScenarioConfig {
	std::string name;
	int N = 0;
	double dx = 0.0, dt = 0.0;
	int i_interface = -1;
	double eps_inf_right = 1.0, eps_s_right = 1.0, omega0_right = 1.0,
		   delta_right = 0.0;
	int i_src = 0;
	double src_amp = 1.0, src_t0 = 0.0, src_tau = 1.0, src_f0 = 0.0,
		   src_phase = 0.0;
	double v_left = phys::C0, v_right = phys::C0;
	std::vector<SnapshotSpec> snapshots;
	long nsteps = 0;
	double x_src_m = 0.0;
	double x_d_m = std::numeric_limits<double>::quiet_NaN();
	double L_m = 0.0;
};

constexpr double FS = 1.0e-15;
constexpr double UM = 1.0e-6;

int round_index(double x, double dx) {
	return static_cast<int>(std::llround(x / dx));
}

long steps_for(double total_time_s, double dt) {
	return static_cast<long>(std::ceil(total_time_s / dt));
}

std::complex<double> lorentz_eps(double omega, double eps_inf, double eps_s,
                                 double omega0, double delta) {
	const std::complex<double> j(0.0, 1.0);
	return eps_inf +
	       omega0 * omega0 * (eps_s - eps_inf) /
	               (omega0 * omega0 - j * delta * omega - omega * omega);
}

double lorentz_index(double omega, double eps_inf, double eps_s, double omega0,
                     double delta) {
	return std::sqrt(lorentz_eps(omega, eps_inf, eps_s, omega0, delta)).real();
}

double lorentz_group_index(double omega, double eps_inf, double eps_s,
                           double omega0, double delta) {
	const double h = 1.0e-6 * omega;
	const double n = lorentz_index(omega, eps_inf, eps_s, omega0, delta);
	const double dn = lorentz_index(omega + h, eps_inf, eps_s, omega0, delta) -
	                  lorentz_index(omega - h, eps_inf, eps_s, omega0, delta);
	return n + omega * dn / (2.0 * h);
}

ScenarioConfig make_scenario(const std::string& name,
                             std::optional<double> eps_s_override,
                             std::optional<long> nsteps_override) {
	ScenarioConfig cfg;
	cfg.name = name;
	cfg.dx = 2.0e-8;
	cfg.dt = 0.5 * cfg.dx / phys::C0;

	if (name == "vacuum") {
		if (eps_s_override) {
			throw std::invalid_argument("--eps_s has no effect for vacuum");
		}
		cfg.L_m = 30.0 * UM;
		cfg.N = round_index(cfg.L_m, cfg.dx);
		cfg.i_interface = -1;
		cfg.x_src_m = 4.0 * UM;
		cfg.i_src = round_index(cfg.x_src_m, cfg.dx);
		cfg.src_amp = 1.0;
		cfg.src_tau = 5.0 * FS;
		cfg.src_t0 = 5.0 * cfg.src_tau;
		cfg.src_f0 = 0.0;
		cfg.src_phase = 0.0;
		cfg.v_left = phys::C0;
		cfg.v_right = phys::C0;
		for (double t_fs : {45.0, 55.0, 65.0, 75.0, 85.0, 95.0}) {
			cfg.snapshots.push_back(
					{t_fs * FS,
					 "t" + std::to_string(static_cast<int>(t_fs)) + "fs"});
		}
		cfg.nsteps = steps_for(95.0 * FS, cfg.dt);
	} else if (name == "dielectric") {
		cfg.L_m = 34.0 * UM;
		cfg.N = round_index(cfg.L_m, cfg.dx);
		cfg.x_d_m = 14.0 * UM;
		cfg.i_interface = round_index(cfg.x_d_m, cfg.dx);
		cfg.eps_inf_right = eps_s_override.value_or(2.25);
		cfg.eps_s_right = cfg.eps_inf_right;
		if (cfg.eps_inf_right < 1.0) {
			throw std::invalid_argument(
					"--eps_s must be >= 1 for a dielectric");
		}
		cfg.omega0_right = 1.0;
		cfg.delta_right = 0.0;
		cfg.x_src_m = 4.0 * UM;
		cfg.i_src = round_index(cfg.x_src_m, cfg.dx);
		cfg.src_amp = 1.0;
		cfg.src_tau = 5.0 * FS;
		cfg.src_t0 = 5.0 * cfg.src_tau;
		cfg.src_f0 = 0.0;
		cfg.src_phase = 0.0;
		cfg.v_left = phys::C0;
		cfg.v_right = phys::C0 / std::sqrt(cfg.eps_inf_right);
		cfg.snapshots = {
				{38.36 * FS, "approaching_interface"},
				{58.36 * FS, "at_interface"},
				{78.36 * FS, "splitting"},
				{93.36 * FS, "separated"},
		};
		cfg.nsteps = steps_for(108.36 * FS, cfg.dt);
	} else if (name == "lorentz") {
		cfg.x_d_m = 45.0 * UM;
		cfg.i_interface = round_index(cfg.x_d_m, cfg.dx);
		cfg.eps_inf_right = 2.25;
		cfg.eps_s_right = eps_s_override.value_or(5.25);
		cfg.omega0_right = 4.0e14;
		cfg.delta_right = 2.0e9;
		cfg.x_src_m = 10.0 * UM;
		cfg.i_src = round_index(cfg.x_src_m, cfg.dx);
		cfg.src_amp = 1.0;
		cfg.src_tau = 15.0 * FS;
		cfg.src_t0 = 5.0 * cfg.src_tau;
		cfg.src_f0 = 1.37e14;
		cfg.src_phase = 0.0;
		const double omega_c = 2.0 * phys::PI * cfg.src_f0;
		if (cfg.eps_s_right < cfg.eps_inf_right) {
			throw std::invalid_argument(
					"--eps_s must be >= eps_inf = 2.25 for a passive Lorentz "
					"medium");
		}
		if (lorentz_eps(omega_c, cfg.eps_inf_right, cfg.eps_s_right,
		                cfg.omega0_right, cfg.delta_right)
		            .real() <= 0.0) {
			throw std::invalid_argument(
					"--eps_s puts the carrier frequency in the stop band");
		}
		const double n_c =
				lorentz_index(omega_c, cfg.eps_inf_right, cfg.eps_s_right,
				              cfg.omega0_right, cfg.delta_right);
		const double n_g =
				lorentz_group_index(omega_c, cfg.eps_inf_right, cfg.eps_s_right,
				                    cfg.omega0_right, cfg.delta_right);
		cfg.v_left = phys::C0;
		cfg.v_right = phys::C0 / n_c;
		const double t_at = cfg.src_t0 + (cfg.x_d_m - cfg.x_src_m) / phys::C0;
		cfg.snapshots = {
				{120.0 * FS, "approaching_interface"},
				{t_at, "at_interface"},
				{t_at + 20.0 * UM * n_g / phys::C0, "depth_20um"},
				{t_at + 40.0 * UM * n_g / phys::C0, "depth_40um"},
		};
		cfg.nsteps = steps_for(cfg.snapshots.back().t_s, cfg.dt);
		const double v_fast = phys::C0 / std::sqrt(cfg.eps_inf_right);
		const double t_enter = t_at - 4.0 * cfg.src_tau;
		cfg.L_m =
				std::max(125.0 * UM,
				         cfg.x_d_m + 10.0 * UM +
				                 v_fast * (cfg.snapshots.back().t_s - t_enter));
		cfg.N = round_index(cfg.L_m, cfg.dx);
	} else {
		throw std::invalid_argument("unknown scenario: " + name);
	}

	if (nsteps_override) cfg.nsteps = *nsteps_override;
	return cfg;
}

std::optional<std::string> parse_flag(const std::string& arg,
                                      const std::string& flag) {
	const std::string prefix = flag + "=";
	if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
	return std::nullopt;
}

std::optional<double> to_double(const std::string& s) {
	try {
		std::size_t pos = 0;
		const double v = std::stod(s, &pos);
		if (pos != s.size() || !std::isfinite(v)) return std::nullopt;
		return v;
	} catch (const std::exception&) {
		return std::nullopt;
	}
}

std::optional<long> to_long(const std::string& s) {
	try {
		std::size_t pos = 0;
		const long v = std::stol(s, &pos);
		if (pos != s.size()) return std::nullopt;
		return v;
	} catch (const std::exception&) {
		return std::nullopt;
	}
}

constexpr const char* kUsage =
		"usage: fdtd1d --scenario=vacuum|dielectric|lorentz "
		"[--outdir=output] [--eps_s=<value>] [--nsteps=<int>]\n";

void remove_old_output(const fs::path& dir) {
	fs::remove(dir / "index.csv");
	for (const auto& entry : fs::directory_iterator(dir)) {
		const std::string name = entry.path().filename().string();
		if (name.rfind("snapshot_", 0) == 0 &&
		    entry.path().extension() == ".csv") {
			fs::remove(entry.path());
		}
	}
}

int run(const std::string& scenario_name, const std::string& outdir,
        std::optional<double> eps_s_override,
        std::optional<long> nsteps_override) {
	ScenarioConfig cfg =
			make_scenario(scenario_name, eps_s_override, nsteps_override);

	Fdtd1D grid(cfg.N, cfg.dx, cfg.dt);
	if (cfg.i_interface >= 0) {
		grid.set_material_region(0, cfg.i_interface - 1, 1.0, 1.0, 1.0, 0.0);
		grid.set_material_region(cfg.i_interface, cfg.N, cfg.eps_inf_right,
		                         cfg.eps_s_right, cfg.omega0_right,
		                         cfg.delta_right);
	} else {
		grid.set_material_region(0, cfg.N, 1.0, 1.0, 1.0, 0.0);
	}
	grid.set_source(cfg.i_src, cfg.src_amp, cfg.src_t0, cfg.src_tau, cfg.src_f0,
	                cfg.src_phase);
	grid.set_abc(cfg.v_left, cfg.v_right);

	const fs::path scenario_dir = fs::path(outdir) / cfg.name;
	fs::create_directories(scenario_dir);
	remove_old_output(scenario_dir);

	struct RecordedSnapshot {
		int index;
		std::string filename;
		double t_s;
		std::string label;
	};
	std::vector<RecordedSnapshot> recorded;

	std::size_t next_snapshot = 0;
	for (long step = 1; step <= cfg.nsteps; ++step) {
		grid.step();
		while (next_snapshot < cfg.snapshots.size() &&
		       grid.time() >= cfg.snapshots[next_snapshot].t_s - cfg.dt / 2.0) {
			char name_buf[32];
			std::snprintf(name_buf, sizeof(name_buf), "snapshot_%02zu.csv",
			              next_snapshot);
			const fs::path snap_path = scenario_dir / name_buf;
			grid.write_snapshot_csv(snap_path.string(), cfg.name);
			recorded.push_back({static_cast<int>(next_snapshot), name_buf,
			                    grid.time(),
			                    cfg.snapshots[next_snapshot].label});
			++next_snapshot;
		}
	}

	for (double e : grid.ez()) {
		if (!std::isfinite(e)) throw std::runtime_error("solution diverged");
	}
	if (recorded.size() < cfg.snapshots.size()) {
		std::cerr << "warning: only " << recorded.size() << " of "
		          << cfg.snapshots.size()
		          << " snapshots recorded; increase --nsteps\n";
	}

	const fs::path index_path = scenario_dir / "index.csv";
	std::ofstream index_out(index_path);
	if (!index_out) {
		throw std::runtime_error("could not open " + index_path.string());
	}
	index_out << std::scientific << std::setprecision(6);
	index_out << "# scenario=" << cfg.name << " dx_m=" << cfg.dx
	          << " dt_s=" << cfg.dt << " x_src_m=" << cfg.x_src_m << " x_d_m=";
	if (std::isnan(cfg.x_d_m)) {
		index_out << "NA";
	} else {
		index_out << cfg.x_d_m;
	}
	index_out << " L_m=" << cfg.L_m << " eps_inf_right=" << cfg.eps_inf_right
	          << " eps_s_right=" << cfg.eps_s_right
	          << " omega0_right=" << cfg.omega0_right
	          << " delta_right=" << cfg.delta_right
	          << " src_f0_hz=" << cfg.src_f0 << "\n";
	index_out << "index,filename,t_s,t_fs,label\n";
	index_out << std::defaultfloat;
	for (const auto& r : recorded) {
		index_out << r.index << "," << r.filename << "," << std::scientific
		          << std::setprecision(6) << r.t_s << "," << std::defaultfloat
		          << std::setprecision(4) << (r.t_s / FS) << "," << r.label
		          << "\n";
	}
	index_out.close();
	if (!index_out) {
		throw std::runtime_error("could not write " + index_path.string());
	}

	std::cout << "scenario=" << cfg.name << " steps=" << cfg.nsteps
	          << " snapshots=" << recorded.size() << " -> "
	          << scenario_dir.string() << "\n";
	return 0;
}

}

int main(int argc, char** argv) {
	std::string scenario_name;
	std::string outdir = "output";
	std::optional<double> eps_s_override;
	std::optional<long> nsteps_override;

	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h") {
			std::cout << kUsage;
			return 0;
		} else if (auto v = parse_flag(arg, "--scenario")) {
			scenario_name = *v;
		} else if (auto v = parse_flag(arg, "--outdir")) {
			outdir = *v;
		} else if (auto v = parse_flag(arg, "--eps_s")) {
			eps_s_override = to_double(*v);
			if (!eps_s_override || *eps_s_override <= 0.0) {
				std::cerr << "--eps_s must be a positive number, got '" << *v
				          << "'\n";
				return 1;
			}
		} else if (auto v = parse_flag(arg, "--nsteps")) {
			nsteps_override = to_long(*v);
			if (!nsteps_override || *nsteps_override <= 0) {
				std::cerr << "--nsteps must be a positive integer, got '" << *v
				          << "'\n";
				return 1;
			}
		} else {
			std::cerr << "unrecognized argument: " << arg << "\n" << kUsage;
			return 1;
		}
	}

	if (scenario_name != "vacuum" && scenario_name != "dielectric" &&
	    scenario_name != "lorentz") {
		std::cerr << kUsage;
		return 1;
	}
	if (outdir.empty()) {
		std::cerr << "--outdir must not be empty\n";
		return 1;
	}

	try {
		return run(scenario_name, outdir, eps_s_override, nsteps_override);
	} catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
		return 1;
	}
}
