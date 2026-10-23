import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FDTD1D = Path(os.environ.get("FDTD1D_BIN", ROOT / "build" / "fdtd1d"))

pytestmark = pytest.mark.skipif(not FDTD1D.exists(),
                                reason=f"solver binary not found: {FDTD1D}")


def run(*args):
    return subprocess.run([str(FDTD1D), *args], capture_output=True, text=True)


@pytest.mark.parametrize("scenario", ["vacuum", "dielectric", "lorentz"])
def test_scenario_runs(scenario, tmp_path):
    res = run(f"--scenario={scenario}", f"--outdir={tmp_path}")
    assert res.returncode == 0, res.stderr
    assert (tmp_path / scenario / "index.csv").exists()


@pytest.mark.parametrize("arg", ["--scenario=bogus", "--bogus", "--nsteps=1e3",
                                 "--nsteps=0", "--eps_s=nan", "--outdir=",
                                 "--eps_s=1.0", "--eps_s=20"])
def test_invalid_argument_is_rejected(arg, tmp_path):
    res = run("--scenario=lorentz", f"--outdir={tmp_path}", arg)
    assert res.returncode != 0
    assert res.stderr


def test_eps_s_is_rejected_for_vacuum(tmp_path):
    res = run("--scenario=vacuum", f"--outdir={tmp_path}", "--eps_s=4")
    assert res.returncode != 0
    assert "vacuum" in res.stderr


@pytest.mark.parametrize("eps", ["0.1", "0.5"])
def test_dielectric_rejects_permittivity_below_one(eps, tmp_path):
    res = run("--scenario=dielectric", f"--outdir={tmp_path}", f"--eps_s={eps}")
    assert res.returncode != 0
    assert ">= 1" in res.stderr


def test_rerun_removes_stale_output(tmp_path):
    out = tmp_path / "vacuum"
    out.mkdir()
    (out / "index.csv").write_text("stale\n")
    (out / "snapshot_42.csv").write_text("stale\n")
    res = run("--scenario=vacuum", f"--outdir={tmp_path}", "--nsteps=10")
    assert res.returncode == 0
    assert not (out / "snapshot_42.csv").exists()
    assert (out / "index.csv").read_text() != "stale\n"


def test_help():
    res = run("--help")
    assert res.returncode == 0
    assert "usage:" in res.stdout
