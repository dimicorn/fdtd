#!/usr/bin/env python3
import argparse
import cmath
import csv
import math
import sys
from pathlib import Path

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * math.pi
C0 = 1.0 / math.sqrt(EPS0 * MU0)


def read_index(outdir):
    path = Path(outdir) / "index.csv"
    meta = {}
    rows = []
    with open(path) as f:
        comment = f.readline().strip()
        for tok in comment.lstrip("#").split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                meta[k] = v
        reader = csv.DictReader(f)
        rows = list(reader)
    return meta, rows


def read_snapshot(outdir, filename):
    path = Path(outdir) / filename
    xs, es = [], []
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f:
            x, e = line.strip().split(",")
            xs.append(float(x))
            es.append(float(e))
    if not all(math.isfinite(e) for e in es):
        raise ValueError(f"{path}: non-finite Ez values")
    return xs, es


def window(xs, es, x_lo=None, x_hi=None):
    pairs = [(x, e) for x, e in zip(xs, es)
             if (x_lo is None or x >= x_lo) and (x_hi is None or x <= x_hi)]
    if not pairs:
        raise ValueError(f"no samples in window [{x_lo}, {x_hi}]")
    return [x for x, _ in pairs], [e for _, e in pairs]


def energy(xs, es, x_lo=None, x_hi=None):
    xw, ew = window(xs, es, x_lo, x_hi)
    dx = xs[1] - xs[0]
    return sum(e * e for e in ew) * dx


def centroid(xs, es, x_lo=None, x_hi=None):
    xw, ew = window(xs, es, x_lo, x_hi)
    den = sum(e * e for e in ew)
    if den == 0.0:
        raise ValueError(f"zero field in window [{x_lo}, {x_hi}]")
    return sum(x * e * e for x, e in zip(xw, ew)) / den


def rms_width(xs, es, x_lo=None, x_hi=None):
    xc = centroid(xs, es, x_lo, x_hi)
    xw, ew = window(xs, es, x_lo, x_hi)
    den = sum(e * e for e in ew)
    return math.sqrt(sum((x - xc) ** 2 * e * e for x, e in zip(xw, ew)) / den)


def peak_abs(xs, es, x_lo=None, x_hi=None):
    xw, ew = window(xs, es, x_lo, x_hi)
    if not all(math.isfinite(e) for e in ew):
        raise ValueError("non-finite Ez values")
    i = max(range(len(ew)), key=lambda k: abs(ew[k]))
    return xw[i], ew[i]


def rel_err(a, b):
    return abs(a - b) / abs(b) if b != 0 else float("inf")


def meta_float(meta, key):
    v = meta.get(key)
    return None if v in (None, "NA") else float(v)


def lorentz_index(omega, eps_inf, eps_s, omega0, delta):
    eps = eps_inf + omega0 ** 2 * (eps_s - eps_inf) / (
        omega0 ** 2 - 1j * delta * omega - omega ** 2)
    return cmath.sqrt(eps)


def group_index(omega, *params):
    h = 1e-6 * omega
    n = lorentz_index(omega, *params).real
    dn = (lorentz_index(omega + h, *params).real -
          lorentz_index(omega - h, *params).real)
    return n + omega * dn / (2.0 * h)


def spectral_group_velocity(xs, es, params):
    eps_inf, eps_s, omega0, _ = params
    dx = xs[1] - xs[0]
    power = np.abs(np.fft.rfft(es)) ** 2
    k = 2.0 * math.pi * np.fft.rfftfreq(len(es), dx)
    omega_l = omega0 * math.sqrt(eps_s / eps_inf)
    omegas = np.linspace(omega_l * 1.0001, 20.0 * omega_l, 20000)
    k_of_omega = np.array([w * lorentz_index(w, *params).real / C0
                           for w in omegas])
    if not np.all(np.diff(k_of_omega) > 0):
        raise ValueError("k(omega) is not monotonic for this medium")
    vg_of_omega = np.array([C0 / group_index(w, *params) for w in omegas])
    sel = (k > k_of_omega[0]) & (k < k_of_omega[-1])
    vg = np.interp(k[sel], k_of_omega, vg_of_omega)
    return float((power[sel] * vg).sum() / power[sel].sum())


def cmd_free_space(args):
    meta, rows = read_index(args.outdir)
    if len(rows) < 2:
        print(f"free-space check: need at least 2 snapshots, found {len(rows)}")
        return False
    x_src = meta_float(meta, "x_src_m")
    ts, xcs = [], []
    for row in rows:
        xs, es = read_snapshot(args.outdir, row["filename"])
        ts.append(float(row["t_s"]))
        xcs.append(centroid(xs, es, x_lo=x_src))

    n = len(ts)
    t_mean = sum(ts) / n
    x_mean = sum(xcs) / n
    num = sum((t - t_mean) * (x - x_mean) for t, x in zip(ts, xcs))
    den = sum((t - t_mean) ** 2 for t in ts)
    slope = num / den

    tol = 0.001
    err = rel_err(slope, C0)
    ok = err < tol
    print(f"free-space check ({args.outdir}):")
    print(f"  fitted speed = {slope:.6e} m/s")
    print(f"  C0           = {C0:.6e} m/s")
    print(f"  rel. error   = {err*100:.4f}%  -> "
          f"{'PASS' if ok else 'FAIL'} (tolerance {tol*100:g}%)")
    return ok


def cmd_fresnel(args):
    meta, rows = read_index(args.outdir)
    x_d = meta_float(meta, "x_d_m")
    if x_d is None:
        print(f"fresnel check: {args.outdir} has no interface (x_d_m=NA)")
        return False
    x_src = meta_float(meta, "x_src_m")
    by_label = {row["label"]: row for row in rows}

    approaching = by_label.get(args.incident_label)
    separated = by_label.get(args.after_label)
    if approaching is None or separated is None:
        print(f"fresnel check: required snapshots not found "
              f"(need '{args.incident_label}' and '{args.after_label}')")
        return False

    eps_r = args.eps_r
    eps_inf = meta_float(meta, "eps_inf_right")
    eps_s = meta_float(meta, "eps_s_right")
    if eps_r is None:
        if eps_inf is None or eps_s != eps_inf:
            print("fresnel check: --eps-r is required for a dispersive medium")
            return False
        eps_r = eps_inf
    elif eps_inf is not None and eps_s == eps_inf and eps_r != eps_inf:
        print(f"  note: --eps-r={eps_r} differs from the run's "
              f"eps_inf_right=eps_s_right={eps_inf}")

    xs, es = read_snapshot(args.outdir, approaching["filename"])
    w_incident = energy(xs, es, x_lo=x_src, x_hi=x_d)
    _, e_incident = peak_abs(xs, es, x_lo=x_src, x_hi=x_d)

    xs2, es2 = read_snapshot(args.outdir, separated["filename"])
    w_reflected = energy(xs2, es2, x_hi=x_d)
    w_transmitted = energy(xs2, es2, x_lo=x_d)
    _, e_reflected = peak_abs(xs2, es2, x_hi=x_d)

    n = math.sqrt(eps_r)
    r_analytic = (1.0 - n) / (1.0 + n)
    t_analytic = 2.0 / (1.0 + n)

    sign = 1.0 if e_reflected * e_incident >= 0 else -1.0
    r_meas = sign * math.sqrt(w_reflected / w_incident)
    t_meas = math.sqrt(n * w_transmitted / w_incident)
    R_meas = r_meas ** 2
    T_meas = n * t_meas ** 2

    tol = 0.002
    err_r, err_t = abs(r_meas - r_analytic), abs(t_meas - t_analytic)
    err_rt = abs(R_meas + T_meas - 1.0)
    print(f"fresnel check ({args.outdir}, eps_r={eps_r}, n={n:.4f}):")
    print(f"  r_meas={r_meas:+.4f}  r_analytic={r_analytic:+.4f}  "
          f"abs.err={err_r:.4f}")
    print(f"  t_meas={t_meas:+.4f}  t_analytic={t_analytic:+.4f}  "
          f"abs.err={err_t:.4f}")
    print(f"  R_meas={R_meas:.4f}  T_meas={T_meas:.4f}  "
          f"R+T={R_meas+T_meas:.4f} (should be ~1.0)")
    ok = err_r < tol and err_t < tol and err_rt < tol
    print(f"  -> {'PASS' if ok else 'FAIL'} "
          f"(tolerance {tol:g} of the incident amplitude)")
    return ok


def cmd_dispersion(args):
    meta, rows = read_index(args.outdir)
    by_label = {row["label"]: row for row in rows}
    a, b = by_label.get("depth_20um"), by_label.get("depth_40um")
    if a is None or b is None:
        print("dispersion check: required snapshots not found "
              "(need 'depth_20um' and 'depth_40um')")
        return False
    keys = ("x_d_m", "eps_inf_right", "eps_s_right", "omega0_right",
            "delta_right", "src_f0_hz")
    vals = [meta_float(meta, k) for k in keys]
    if None in vals:
        print(f"dispersion check: index.csv lacks metadata {keys}")
        return False
    x_d, eps_inf, eps_s, omega0, delta, f0 = vals

    xs1, es1 = read_snapshot(args.outdir, a["filename"])
    xs2, es2 = read_snapshot(args.outdir, b["filename"])
    x1, x2 = centroid(xs1, es1, x_lo=x_d), centroid(xs2, es2, x_lo=x_d)
    w1, w2 = rms_width(xs1, es1, x_lo=x_d), rms_width(xs2, es2, x_lo=x_d)
    t1, t2 = float(a["t_s"]), float(b["t_s"])

    params = (eps_inf, eps_s, omega0, delta)
    n_g = group_index(2.0 * math.pi * f0, *params)
    xw, ew = window(xs1, es1, x_lo=x_d)
    v_analytic = spectral_group_velocity(xw, ew, params)
    v_meas = (x2 - x1) / (t2 - t1)
    tol = 0.005
    err = rel_err(v_meas, v_analytic)
    ok = err < tol
    growth = (w2 - w1) / w1 * 100.0
    print(f"dispersion check ({args.outdir}):")
    print(f"  centroid depth at depth_20um = {(x1 - x_d)*1e6:.3f} um")
    print(f"  centroid depth at depth_40um = {(x2 - x_d)*1e6:.3f} um")
    print(f"  group velocity: measured {v_meas:.6e} m/s, "
          f"analytic (spectral mean) {v_analytic:.6e} m/s")
    print(f"  (at the carrier: n_g={n_g:.4f}, v_g={C0/n_g:.6e} m/s)")
    print(f"  rel. error = {err*100:.3f}%  -> "
          f"{'PASS' if ok else 'FAIL'} (tolerance {tol*100:g}%)")
    print(f"  RMS width at depth_20um = {w1*1e6:.4f} um")
    print(f"  RMS width at depth_40um = {w2*1e6:.4f} um")
    print(f"  growth = {growth:+.2f}%  (informational)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("free-space")
    p1.add_argument("--outdir", required=True)
    p1.set_defaults(func=cmd_free_space)

    p2 = sub.add_parser("fresnel")
    p2.add_argument("--outdir", required=True)
    p2.add_argument("--eps-r", type=float, default=None,
                    help="default: the run's eps_inf_right, if non-dispersive")
    p2.add_argument("--incident-label", default="approaching_interface")
    p2.add_argument("--after-label", default="separated")
    p2.set_defaults(func=cmd_fresnel)

    p3 = sub.add_parser("dispersion")
    p3.add_argument("--outdir", required=True)
    p3.set_defaults(func=cmd_dispersion)

    args = parser.parse_args()
    try:
        ok = args.func(args)
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError) as e:
        print(f"{args.cmd} check: error: {e}")
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
