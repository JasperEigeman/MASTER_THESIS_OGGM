#!/usr/bin/env python
"""
Recover the VAS reconstruction from glacier folders that are already prepared.

Use case: the all-regions notebook ran for a long time (masks, CRU climate, VAS calibration all
finished) but the reconstruction step produced no results. Everything expensive is already on
disk, so this script does NOT redo it. It only
  1. inventories the existing glacier folders (which files exist, how many glaciers are valid),
  2. PROBES the reconstruction on a few glaciers with errors switched ON (the notebook runs with
     continue_on_error=True, which hides the real error in each glacier's log.txt),
  3. if the probe works: runs the missing reconstruction for all glaciers of the region,
     reads the real reason for every failed glacier from its log.txt, writes the regional
     results / plots / summary exactly like the notebook, and validates them.

It reuses the definitions of the notebook itself (cells 1-11: settings, OGGM setup, helper
functions), so all settings are identical to the notebook run. The notebook's RUN cell is not executed.

Examples
  export OGGM_WORK_ROOT=/path/to/oggm_work           # same value as in the original run
  python recover_vas_reconstruction.py NOTEBOOK.ipynb --probe-only --regions 11
  python recover_vas_reconstruction.py NOTEBOOK.ipynb --regions 11
  python recover_vas_reconstruction.py NOTEBOOK.ipynb --regions 1-19 --clean
"""
import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

# --- NumPy 2 compatibility -------------------------------------------------------------------
# Older OGGM / OGGM-VAS code still uses names that NumPy 2.0 removed (np.NaN, np.Inf, ...), which made
# every glacier fail in the reconstruction run. Re-create the removed aliases (only if missing).
for _old, _new in {"NaN": np.nan, "Inf": np.inf, "infty": np.inf, "PINF": np.inf, "NINF": -np.inf,
                   "float_": np.float64, "complex_": np.complex128, "product": np.prod,
                   "cumproduct": np.cumprod, "sometrue": np.any, "alltrue": np.all}.items():
    if not hasattr(np, _old):
        setattr(np, _old, _new)

# files a glacier folder needs for the reconstruction run
REQUIRED_FILES = ["climate_historical.nc", "climate_info.json", "gridded_data.nc",
                  "glacier_grid.json", "vascaling_mustar.json"]


# ----------------------------------------------------------------------------------------------
# loading the notebook definitions
# ----------------------------------------------------------------------------------------------
def load_notebook_definitions(nb_path):
    """Execute all code cells of the notebook except the RUN cell (they only define things)."""
    nb = json.load(open(nb_path))
    ns = {"__name__": "recon_notebook"}
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(cells, start=1):
        src = "".join(cell["source"])
        if "RUN_ALL_REGIONS" in src and "REGIONS_TO_RUN" in src:
            continue                                            # the RUN cell
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        exec(compile(src, f"<notebook code cell {i}>", "exec"), ns)
    ns["SHOW_PLOTS"] = False
    needed = ["set_region", "init_oggm", "run_cru_reconstruction", "write_cru_climate", "plot_all",
              "make_summary", "region_signature", "clean_glacier_dirs", "collect_results",
              "GlacierDirectory", "vascaling", "cfg", "WORK_ROOT", "MODEL_START_YEAR",
              "CRU_RECON_END_YEAR", "CRU_HIST_SUFFIX"]
    missing = [n for n in needed if n not in ns]
    if missing:
        sys.exit(f"The notebook does not define {missing}: wrong notebook?")
    return ns


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def parse_regions(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def glacier_ids(per_glacier):
    """All glacier folders: per_glacier/<region>/<region-xx>/<glacier id>."""
    return sorted(p.name for p in per_glacier.glob("*/*/*") if p.is_dir())


def glacier_folder(per_glacier, rgi_id):
    return per_glacier / rgi_id[:-6] / rgi_id[:-3] / rgi_id        # same layout as OGGM


def last_log_line(folder, task="run_historic"):
    """Real error of a glacier from its log.txt (OGGM writes one line per task: ...;task;status)."""
    log = Path(folder) / "log.txt"
    if not log.exists():
        return "no log.txt entry found"
    lines = [l.strip() for l in log.read_text(errors="replace").splitlines() if l.strip()]
    mine = [l for l in lines if task in l]
    line = (mine or lines or ["empty log.txt"])[-1]
    return line.split(";")[-1].strip()[:300] if ";" in line else line[:300]


def inventory(per_glacier, ids, suffix):
    rows = {name: 0 for name in REQUIRED_FILES}
    done = 0
    for rid in ids:
        folder = glacier_folder(per_glacier, rid)
        for name in REQUIRED_FILES:
            rows[name] += (folder / name).exists()
        done += (folder / f"model_diagnostics{suffix}.nc").exists()
    print(f"  glacier folders: {len(ids)}")
    for name, n in rows.items():
        print(f"    {name:26s} {n:7d}  {'OK' if n == len(ids) else 'INCOMPLETE'}")
    print(f"    {'model_diagnostics' + suffix + '.nc':26s} {done:7d}  (reconstruction already done)")
    return done


def probe(ns, gdirs, n):
    """Run the reconstruction on n glaciers with errors ON. True if all worked."""
    cfg = ns["cfg"]
    run = ns["vascaling"].run_historic_from_climate_data
    suffix = ns["CRU_HIST_SUFFIX"]
    try:
        print("  signature of run_historic_from_climate_data:", inspect.signature(run))
    except Exception:
        pass
    kwargs = dict(ys=ns["MODEL_START_YEAR"], ye=ns["CRU_RECON_END_YEAR"], output_filesuffix=suffix,
                  climate_filename="climate_historical")      # same call as in the notebook
    todo = [g for g in gdirs if not os.path.exists(g.get_filepath("model_diagnostics", filesuffix=suffix))]
    sample = todo[:n]
    if not sample:
        print("  probe skipped: every glacier already has its reconstruction file.")
        return True
    old = cfg.PARAMS.get("continue_on_error", True)
    cfg.PARAMS["continue_on_error"] = False                 # show the REAL error
    ok = True
    try:
        for g in sample:
            t0 = time.time()
            try:
                run(g, **kwargs)
                out = g.get_filepath("model_diagnostics", filesuffix=suffix)
                if not os.path.exists(out):
                    raise RuntimeError(f"run finished without error but {out} was not written")
                print(f"  probe {g.rgi_id}: OK ({time.time() - t0:.1f} s)")
            except Exception:
                ok = False
                print(f"  probe {g.rgi_id}: FAILED. Full error:\n")
                traceback.print_exc()
                print(f"\n  last line of its log.txt: {last_log_line(g.dir)}")
                break
    finally:
        cfg.PARAMS["continue_on_error"] = old
    return ok


def classify_failures(gdirs, suffix, workdir):
    """For glaciers without a reconstruction file: the real reason, from their log.txt."""
    rows = []
    for g in gdirs:
        if not os.path.exists(g.get_filepath("model_diagnostics", filesuffix=suffix)):
            rows.append({"rgi_id": g.rgi_id, "reason": last_log_line(g.dir)})
    df = pd.DataFrame(rows, columns=["rgi_id", "reason"])
    df.to_csv(workdir / "failed_reconstruction_reasons.csv", index=False)
    if len(df):
        print(f"  {len(df)} glaciers without reconstruction. Most common real reasons:")
        for reason, n in df["reason"].value_counts().head(5).items():
            example = df.loc[df["reason"] == reason, "rgi_id"].iloc[0]
            print(f"    {n:6d} x {reason}   (e.g. {example})")
    return df


def validate_region(regional, n_valid, n_ok, ns):
    """Checks that the regional result is plausible. Returns a list of (level, message)."""
    y0, y1 = ns["MODEL_START_YEAR"], ns["CRU_RECON_END_YEAR"]
    out = []
    years = regional["year"].astype(int)
    if years.min() != y0 or years.max() != y1 or len(years) != y1 - y0 + 1:
        out.append(("FAIL", f"years {years.min()}-{years.max()} ({len(years)} rows), expected {y0}-{y1}"))
    if regional[["volume_km3", "area_km2"]].isna().any().any():
        out.append(("FAIL", "NaN in regional volume/area"))
    if regional["volume_km3"].iloc[0] <= 0:
        out.append(("FAIL", "start volume is not positive"))
    if regional["volume_km3"].iloc[-1] > 1.05 * regional["volume_km3"].iloc[0]:
        out.append(("WARN", "regional volume grew by more than 5 % over the period"))
    if n_valid and n_ok / n_valid < 0.90:
        out.append(("WARN", f"only {n_ok}/{n_valid} calibrated glaciers were reconstructed "
                            f"({100 * n_ok / n_valid:.1f} %); check failed_reconstruction_reasons.csv"))
    if not out:
        out.append(("PASS", f"{n_ok}/{n_valid} glaciers, {y0}-{y1}, volume "
                            f"{regional['volume_km3'].iloc[0]:.1f} -> {regional['volume_km3'].iloc[-1]:.1f} km3"))
    return out


# ----------------------------------------------------------------------------------------------
# one region
# ----------------------------------------------------------------------------------------------
def recover_region(ns, region, probe_n, probe_only, clean):
    ns["set_region"](region)
    workdir = ns["WORKDIR"]
    suffix = ns["CRU_HIST_SUFFIX"]
    per_glacier = workdir / "per_glacier"
    print("\n" + "#" * 70 + f"\n# REGION {region:02d}: {ns['REGION_NAME']}\n" + "#" * 70)
    print(f"  folder: {workdir}")

    if not per_glacier.exists():
        done = (workdir / "region_done_cru.json").exists()
        print("  no per_glacier folder: " + ("region finished earlier (folders cleaned), nothing to recover."
                                             if done else "nothing prepared here, run the notebook for this region."))
        return {"region": region, "status": "skipped_no_glacier_folders"}

    ids = glacier_ids(per_glacier)
    if not ids:
        print("  per_glacier exists but contains no glacier folders.")
        return {"region": region, "status": "no_glaciers"}
    inventory(per_glacier, ids, suffix)

    ns["init_oggm"]()                                           # same OGGM settings as the notebook
    ns["cfg"].PARAMS["border"] = 10
    gdirs, bad_open = [], []
    for rid in ids:
        try:
            gdirs.append(ns["GlacierDirectory"](rid))
        except Exception as exc:
            bad_open.append((rid, f"{type(exc).__name__}: {exc}"))
    if bad_open:
        print(f"  {len(bad_open)} glacier folders could not be opened, e.g. {bad_open[0]}")
    if not gdirs:
        raise RuntimeError("No glacier folder could be opened with OGGM (see message above).")

    valid = []
    for g in gdirs:                                             # same rule as the notebook
        try:
            cal = g.read_json("vascaling_mustar")
            if np.isfinite([cal.get("t_star", np.nan), cal.get("bias", np.nan),
                         cal.get("mu_star", np.nan)]).all():
                valid.append(g)
        except Exception:
            pass
    print(f"  calibrated glaciers (valid vascaling_mustar.json): {len(valid)} of {len(gdirs)}")
    if not valid:
        raise RuntimeError("No calibrated glaciers: the calibration has to be rerun for this region.")

    print("  --- probe (errors switched on) ---")
    if not probe(ns, valid, probe_n):
        print("\n  STOP: the reconstruction itself fails on the probe glacier(s). The error above is the root cause.")
        return {"region": region, "status": "probe_failed"}
    if probe_only:
        return {"region": region, "status": "probe_ok"}

    print("  --- reconstruction of the missing glaciers ---")
    try:
        ok_gdirs, regional, _ = ns["run_cru_reconstruction"](valid)
    except Exception as exc:
        print(f"  reconstruction produced no usable result: {type(exc).__name__}: {exc}")
        classify_failures(valid, suffix, workdir)
        return {"region": region, "status": "failed", "error": str(exc)}
    fails = classify_failures(valid, suffix, workdir)

    ns["write_cru_climate"](ok_gdirs)
    ns["plot_all"](regional)
    ns["make_summary"](regional, len(ids), len(valid), len(ok_gdirs))

    checks = validate_region(regional, len(valid), len(ok_gdirs), ns)
    for level, msg in checks:
        print(f"  [{level}] {msg}")
    status = "FAIL" if any(l == "FAIL" for l, _ in checks) else ("ok_with_warnings" if any(l == "WARN" for l, _ in checks) else "ok")
    if status != "FAIL":
        json.dump(ns["region_signature"](), open(workdir / "region_done_cru.json", "w"))
        if clean and status == "ok":
            ns["clean_glacier_dirs"]()
    return {"region": region, "status": status, "n_glaciers": len(ids), "n_calibrated": len(valid),
            "n_reconstructed": len(ok_gdirs), "n_failed": len(fails),
            "checks": " | ".join(f"{l}: {m}" for l, m in checks)}


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="the reconstruction notebook that was used for the run (.ipynb)")
    ap.add_argument("--regions", default="1-19", help="e.g. 13 or 1-19 or 8,11 (default: all)")
    ap.add_argument("--probe", type=int, default=3, help="number of glaciers in the probe (default 3)")
    ap.add_argument("--probe-only", action="store_true", help="only probe, do not run the full region")
    ap.add_argument("--clean", action="store_true",
                    help="delete the glacier folders of a region after it passed all checks")
    ap.add_argument("--mp", type=int, default=0,
                    help="number of processes for the reconstruction (default 0 = single process; untested "
                         "with OGGM-VAS, try on one small region first)")
    args = ap.parse_args()

    ns = load_notebook_definitions(args.notebook)
    if args.mp > 1:
        ns["init_oggm"]()
        ns["cfg"].PARAMS["use_multiprocessing"] = True
        ns["cfg"].PARAMS["mp_processes"] = args.mp
    work_root = ns["WORK_ROOT"]
    print(f"Work root: {work_root}\nNotebook : {args.notebook}")

    results = []
    for region in parse_regions(args.regions):
        t0 = time.time()
        try:
            row = recover_region(ns, region, args.probe, args.probe_only, args.clean)
        except Exception as exc:
            traceback.print_exc()
            row = {"region": region, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        row["minutes"] = round((time.time() - t0) / 60, 1)
        results.append(row)
        pd.DataFrame(results).to_csv(work_root / "recovery_log.csv", index=False)

    # overview of ALL finished regions (rebuilt from the per-region summaries)
    parts = [pd.read_csv(p) for p in sorted(work_root.glob("*_region[0-9][0-9]/reconstruction_summary.csv"))]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(work_root / "all_regions_reconstruction.csv", index=False)

    print("\n" + "=" * 70 + "\nRECOVERY SUMMARY\n" + "=" * 70)
    print(pd.DataFrame(results)[[c for c in ["region", "status", "n_glaciers", "n_calibrated",
                                             "n_reconstructed", "n_failed", "minutes"]
                                 if c in pd.DataFrame(results)]].to_string(index=False))
    print(f"\nLog: {work_root / 'recovery_log.csv'}")
    bad = [r for r in results if r["status"] in ("probe_failed", "failed", "error", "FAIL")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
