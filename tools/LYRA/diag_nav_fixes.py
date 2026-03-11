#!/usr/bin/env python3
"""Diagnose Stanford nav file accuracy using Rose 1978 navigation fixes.

Compares Stanford nav positions at known fix points (overhead passes,
terminal positions) against ground truth from Rose (1978) Table 2.3.

Usage:
    python diag_nav_fixes.py              # all 19 flights
    python diag_nav_fixes.py 137          # single flight
    python diag_nav_fixes.py 137 147 148  # multiple flights
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lyra


def main():
    flights = [int(a.lstrip("Ff")) for a in sys.argv[1:]] if len(sys.argv) > 1 else None

    if flights:
        for flt in flights:
            lyra.diagnose_nav_fixes(flt, verbose=True)
            print()
    else:
        results = lyra.diagnose_all_nav_fixes(verbose=True)

        # Summary table
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"{'Flight':>7s}  {'Assessment':>20s}  {'Fixes':>5s}  "
              f"{'Diagnosed':>9s}  {'McMurdo ref':>12s}  Residuals (km)")
        print(f"{'------':>7s}  {'----------':>20s}  {'-----':>5s}  "
              f"{'---------':>9s}  {'-----------':>12s}  --------------")

        for flt in sorted(results.keys()):
            r = results[flt]
            fixes = r.get("fixes", [])
            diagnosed = [f for f in fixes if f.get("residual_km") is not None]
            resids = [f["residual_km"] for f in diagnosed]
            resid_str = ", ".join(f"{v:.2f}" for v in resids) if resids else "-"
            mc_refs = [f.get("mcmurdo_ref", "") for f in diagnosed
                       if f.get("mcmurdo_ref")]
            mc = mc_refs[0] if mc_refs else "-"
            print(f"  F{flt:>4d}  {r['assessment']:>20s}  {r.get('n_fixes',0):>5d}  "
                  f"{len(diagnosed):>9d}  {mc:>12s}  {resid_str}")


if __name__ == "__main__":
    main()
