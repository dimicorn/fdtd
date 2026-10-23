import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / "scripts" / "verify.py"
FDTD1D = Path(os.environ.get("FDTD1D_BIN", ROOT / "build" / "fdtd1d"))

spec = importlib.util.spec_from_file_location("verify", VERIFY)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def write_index(outdir, rows, meta="scenario=test dx_m=2.0e-08 x_d_m=0.000014"):
    with open(outdir / "index.csv", "w") as f:
        f.write(f"# {meta}\n")
        f.write("index,filename,t_s,t_fs,label\n")
        for i, (fname, t_s, label) in enumerate(rows):
            f.write(f"{i},{fname},{t_s:.6e},{t_s / 1e-15:.4g},{label}\n")


def write_snapshot(outdir, fname, xs, es):
    with open(outdir / fname, "w") as f:
        f.write("# scenario=test t_s=0 dx_m=2.0e-08\n")
        f.write("x_m,Ez\n")
        for x, e in zip(xs, es):
            f.write(f"{x:.9e},{e:.9e}\n")


def run_verify(*args):
    return subprocess.run([sys.executable, str(VERIFY), *args],
                          capture_output=True, text=True)


def test_rel_err():
    assert verify.rel_err(1.01, 1.0) == pytest.approx(0.01)
    assert verify.rel_err(-0.2, -0.2) == 0.0
    assert verify.rel_err(1.0, 0.0) == math.inf


def test_peak_abs_picks_signed_extremum():
    xs = [0.0, 1.0, 2.0, 3.0]
    es = [0.1, -0.9, 0.5, 0.8]
    assert verify.peak_abs(xs, es) == (1.0, -0.9)
    assert verify.peak_abs(xs, es, x_lo=2.0) == (3.0, 0.8)
    assert verify.peak_abs(xs, es, x_hi=0.5) == (0.0, 0.1)


def test_read_index_and_snapshot_roundtrip(tmp_path):
    write_index(tmp_path, [("snapshot_00.csv", 4.5e-14, "t45fs")])
    write_snapshot(tmp_path, "snapshot_00.csv", [0.0, 2e-8], [0.25, -1.5])

    meta, rows = verify.read_index(tmp_path)
    assert meta["scenario"] == "test"
    assert float(meta["x_d_m"]) == pytest.approx(14e-6)
    assert rows == [{"index": "0", "filename": "snapshot_00.csv",
                     "t_s": "4.500000e-14", "t_fs": "45", "label": "t45fs"}]

    xs, es = verify.read_snapshot(tmp_path, "snapshot_00.csv")
    assert xs == [0.0, 2e-8]
    assert es == [0.25, -1.5]


def test_free_space_recovers_speed_of_synthetic_pulse(tmp_path, capsys):
    rows = []
    xs = [i * 2e-8 for i in range(2000)]
    for k, t in enumerate([40e-15, 60e-15, 80e-15]):
        xc = 5e-6 + verify.C0 * t
        es = [math.exp(-((x - xc) / 1e-6) ** 2) for x in xs]
        fname = f"snapshot_{k:02d}.csv"
        write_snapshot(tmp_path, fname, xs, es)
        rows.append((fname, t, f"t{k}"))
    write_index(tmp_path, rows)

    verify.cmd_free_space(type("Args", (), {"outdir": str(tmp_path)}))
    assert "PASS" in capsys.readouterr().out


needs_solver = pytest.mark.skipif(not FDTD1D.exists(),
                                  reason=f"solver binary not found: {FDTD1D}")


@pytest.fixture(scope="module")
def solver_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("output")
    for scenario in ("vacuum", "dielectric", "lorentz"):
        subprocess.run([str(FDTD1D), f"--scenario={scenario}",
                        f"--outdir={out}"], check=True, capture_output=True)
    return out


@needs_solver
def test_solver_writes_expected_snapshots(solver_output):
    expected = {"vacuum": 6, "dielectric": 4, "lorentz": 4}
    for scenario, count in expected.items():
        _, rows = verify.read_index(solver_output / scenario)
        assert len(rows) == count
        for row in rows:
            xs, es = verify.read_snapshot(solver_output / scenario, row["filename"])
            assert len(xs) == len(es) > 0
            assert all(math.isfinite(e) for e in es)


@needs_solver
def test_free_space_passes(solver_output):
    res = run_verify("free-space", f"--outdir={solver_output / 'vacuum'}")
    assert res.returncode == 0
    assert "PASS" in res.stdout


@needs_solver
def test_fresnel_passes_for_true_permittivity(solver_output):
    res = run_verify("fresnel", f"--outdir={solver_output / 'dielectric'}",
                     "--eps-r=2.25")
    assert res.returncode == 0
    assert "PASS" in res.stdout


@needs_solver
def test_fresnel_reports_fail_for_wrong_permittivity(solver_output):
    res = run_verify("fresnel", f"--outdir={solver_output / 'dielectric'}",
                     "--eps-r=4.0")
    assert "FAIL" in res.stdout


@needs_solver
def test_dielectric_eps_s_sets_permittivity(tmp_path):
    subprocess.run([str(FDTD1D), "--scenario=dielectric", "--eps_s=4",
                    f"--outdir={tmp_path}"], check=True, capture_output=True)
    res = run_verify("fresnel", f"--outdir={tmp_path / 'dielectric'}")
    assert res.returncode == 0, res.stdout
    assert "n=2.0000" in res.stdout


@needs_solver
@pytest.mark.parametrize("eps", ["1", "1000"])
def test_fresnel_passes_without_and_with_strong_reflection(eps, tmp_path):
    subprocess.run([str(FDTD1D), "--scenario=dielectric", f"--eps_s={eps}",
                    f"--outdir={tmp_path}"], check=True, capture_output=True)
    res = run_verify("fresnel", f"--outdir={tmp_path / 'dielectric'}")
    assert res.returncode == 0, res.stdout


@needs_solver
def test_dispersion_passes_near_stop_band(tmp_path):
    subprocess.run([str(FDTD1D), "--scenario=lorentz", "--eps_s=10",
                    f"--outdir={tmp_path}"], check=True, capture_output=True)
    res = run_verify("dispersion", f"--outdir={tmp_path / 'lorentz'}")
    assert res.returncode == 0, res.stdout


@needs_solver
def test_lorentz_regression_matches_fresnel(tmp_path):
    subprocess.run([str(FDTD1D), "--scenario=lorentz", "--eps_s=2.25",
                    f"--outdir={tmp_path}"], check=True, capture_output=True)
    res = run_verify("fresnel", f"--outdir={tmp_path / 'lorentz'}",
                     "--eps-r=2.25", "--after-label=depth_20um")
    assert "PASS" in res.stdout


@needs_solver
def test_dispersion_reports_broadening(solver_output):
    res = run_verify("dispersion", f"--outdir={solver_output / 'lorentz'}")
    assert res.returncode == 0
    growth = float(res.stdout.split("growth =")[1].split("%")[0])
    assert growth > 10.0


@needs_solver
def test_fresnel_fail_sets_nonzero_exit_code(solver_output):
    res = run_verify("fresnel", f"--outdir={solver_output / 'dielectric'}",
                     "--eps-r=4.0")
    assert res.returncode != 0


def test_missing_snapshots_set_nonzero_exit_code(tmp_path):
    write_index(tmp_path, [])
    res = run_verify("fresnel", f"--outdir={tmp_path}", "--eps-r=2.25")
    assert res.returncode != 0


def test_nan_samples_are_not_ignored(tmp_path):
    xs = [0.0, 1.0, 2.0]
    es = [0.1, math.nan, 0.2]
    with pytest.raises(ValueError):
        verify.peak_abs(xs, es)

    write_snapshot(tmp_path, "snapshot_00.csv", xs, es)
    with pytest.raises(ValueError):
        verify.read_snapshot(tmp_path, "snapshot_00.csv")


@needs_solver
def test_dispersion_fails_for_wrong_medium(solver_output, tmp_path):
    src = solver_output / "lorentz"
    for f in src.iterdir():
        text = f.read_text()
        if f.name == "index.csv":
            text = text.replace("eps_s_right=5.250000e+00",
                                "eps_s_right=5.500000e+00")
        (tmp_path / f.name).write_text(text)
    res = run_verify("dispersion", f"--outdir={tmp_path}")
    assert res.returncode != 0
    assert "FAIL" in res.stdout
