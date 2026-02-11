#!/usr/bin/env python3
"""Generate synthetic observations for GRI30 NetBench using Cantera directly.

- Loads gri30.yaml
- Reads conditions CSV (case_id, T0, P0_atm, phi, t_end)
- Applies truth multipliers (equation-based matching)
- Simulates 0D constant-volume ideal gas reactor (adiabatic) by default
- Computes QoIs:
    - ignition_delay: time of max(dT/dt)
    - T_peak: max(T)
    - CO2_final: X_CO2 at t_end
    - CO_final: X_CO at t_end
- Adds noise and writes observation CSV (case_id, observable, value, sigma)

This script is independent of your platform tasks, so it is useful even before assimilation tasks are implemented.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path

def match_equation(target: str, eqs: list[str]) -> int | None:
    # Exact match first
    if target in eqs:
        return eqs.index(target)
    # Fuzzy: remove spaces
    t = target.replace(" ", "")
    for i, e in enumerate(eqs):
        if e.replace(" ", "") == t:
            return i
    # Fuzzy: containment (last resort)
    for i, e in enumerate(eqs):
        if t in e.replace(" ", ""):
            return i
    return None

def simulate_one(ct, mech: str, T0: float, P0_atm: float, phi: float, t_end: float, multipliers: dict[int,float]):
    gas = ct.Solution(mech)
    # CH4/air mixture based on equivalence ratio phi
    # stoichiometric: CH4 + 2 O2 -> CO2 + 2 H2O (O2:2, N2:7.52)
    # Set mixture with phi:
    # actual O2 = stoich/phi
    O2 = 2.0/phi
    N2 = O2 * 3.76
    gas.TPX = T0, P0_atm*ct.one_atm, {"CH4":1.0, "O2":O2, "N2":N2}

    # Apply multipliers
    for ridx, m in multipliers.items():
        gas.set_multiplier(m, ridx)

    # Reactor + network
    r = ct.IdealGasReactor(gas, energy='on')
    net = ct.ReactorNet([r])

    t = 0.0
    dt = min(1e-4, t_end/2000)  # adaptive-ish small step cap
    Ts = []
    ts = []
    lastT = r.T
    lastt = 0.0
    dTdt_max = -1e99
    t_ign = None

    while t < t_end:
        t = net.step()
        Ts.append(r.T)
        ts.append(t)
        # finite difference for dT/dt
        dTdt = (r.T - lastT) / max(1e-12, (t - lastt))
        if dTdt > dTdt_max:
            dTdt_max = dTdt
            t_ign = t
        lastT = r.T
        lastt = t
        # safety break (if solver stalls)
        if len(ts) > 200000:
            break

    # QoIs
    T_peak = max(Ts) if Ts else r.T
    X = r.thermo.X
    sp = r.thermo.species_names
    X_CO2 = X[sp.index("CO2")] if "CO2" in sp else float("nan")
    X_CO  = X[sp.index("CO")]  if "CO" in sp else float("nan")
    return {
        "ignition_delay": float(t_ign) if t_ign is not None else float("nan"),
        "T_peak": float(T_peak),
        "CO2_final": float(X_CO2),
        "CO_final": float(X_CO),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--truth", required=True, help="JSON file with equation->multiplier list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise_x", type=float, default=0.03, help="relative noise for mole fraction")
    ap.add_argument("--noise_T", type=float, default=10.0, help="absolute noise for temperature (K)")
    ap.add_argument("--noise_tau", type=float, default=0.05, help="relative noise for ignition delay")
    args = ap.parse_args()

    import cantera as ct

    mech = args.mechanism
    cond_path = Path(args.conditions)
    truth_path = Path(args.truth)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    items = truth["items"]

    gas = ct.Solution(mech)
    eqs = gas.reaction_equations()
    multipliers = {}
    unresolved = []
    for it in items:
        ridx = match_equation(it["equation"], eqs)
        if ridx is None:
            unresolved.append(it["equation"])
        else:
            multipliers[ridx] = float(it["multiplier"])

    if unresolved:
        print("[WARN] Some truth equations could not be matched. They will be ignored:")
        for e in unresolved:
            print(" -", e)

    rnd = random.Random(args.seed)

    obs_rows = []
    with cond_path.open("r", encoding="utf-8") as f:
        cr = csv.DictReader(f)
        for r in cr:
            cid = r["case_id"]
            T0 = float(r["T0"])
            P0 = float(r["P0_atm"])
            phi = float(r["phi"])
            t_end = float(r["t_end"])
            q = simulate_one(ct, mech, T0, P0, phi, t_end, multipliers)

            # noise add
            # mole fractions
            for k in ["CO2_final","CO_final"]:
                val = q[k]
                sigma = max(1e-12, args.noise_x * max(val, 1e-6))
                noisy = val + rnd.gauss(0.0, sigma)
                obs_rows.append({"case_id": cid, "observable": k, "value": noisy, "sigma": sigma})
            # temperature
            valT = q["T_peak"]
            sigmaT = args.noise_T
            obs_rows.append({"case_id": cid, "observable": "T_peak", "value": valT + rnd.gauss(0.0, sigmaT), "sigma": sigmaT})
            # ignition delay
            tau = q["ignition_delay"]
            sigtau = max(1e-9, args.noise_tau * max(tau, 1e-6))
            obs_rows.append({"case_id": cid, "observable": "ignition_delay", "value": tau + rnd.gauss(0.0, sigtau), "sigma": sigtau})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id","observable","value","sigma"])
        w.writeheader()
        w.writerows(obs_rows)

    print(f"[OK] wrote observations: {out_path}")
    print(f"[INFO] used {len(multipliers)} matched truth multipliers (out of {len(items)})")

if __name__ == "__main__":
    main()
