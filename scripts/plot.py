#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from verify import read_index, read_snapshot

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                         choices=["vacuum", "dielectric", "lorentz"])
    parser.add_argument("--outdir", default=None,
                         help="defaults to <project>/output/<scenario>")
    parser.add_argument("--save", default=None,
                         help="defaults to <outdir>/plot_<scenario>.png")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else ROOT / "output" / args.scenario
    meta, rows = read_index(outdir)

    fig, ax = plt.subplots(figsize=(10, 5))

    x_d = meta.get("x_d_m", "NA")
    if x_d != "NA":
        x_d_um = float(x_d) * 1e6
        L_um = float(meta["L_m"]) * 1e6
        eps_inf = float(meta["eps_inf_right"])
        eps_s = float(meta["eps_s_right"])
        ax.axvspan(x_d_um, L_um, color="tab:blue", alpha=0.12,
                   label=f"dielectric ($\\varepsilon_\\infty$={eps_inf:g}, "
                         f"$\\varepsilon_s$={eps_s:g})")

    for row in rows:
        xs, es = read_snapshot(outdir, row["filename"])
        xs_um = [x * 1e6 for x in xs]
        t_fs = float(row["t_fs"])
        ax.plot(xs_um, es, linewidth=1.2,
                label=f"t={t_fs:.1f} fs ({row['label']})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("Ez (a.u.)")
    ax.set_title(f"1D FDTD: scenario '{args.scenario}'")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    save_path = Path(args.save) if args.save else outdir / f"plot_{args.scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
