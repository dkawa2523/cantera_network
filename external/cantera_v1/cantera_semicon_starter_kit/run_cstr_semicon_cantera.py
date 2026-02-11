# run_cstr_semicon_cantera.py
# pip install cantera matplotlib numpy
import cantera as ct
import numpy as np

# ====== User settings ======
MECH = "plasma_semicon_neutral.yaml"  # ck2yaml output
filter_electron_in_yaml = False       # E/e-/PHOTON reactions removal after YAML load
T = 500.0                             # K
P = 40.0 * ct.one_atm / 760.0         # 40 mTorr
V = 1.0e-4                            # m^3
tau = 0.05                            # s
t_end = 10 * tau

CASE = "CF4_O2_N2"                    # or "CHF3_C2F6_C4F8_AIR"
if CASE == "CF4_O2_N2":
    X_in = {"CF4":0.40, "O2":0.40, "N2":0.20}
elif CASE == "CHF3_C2F6_C4F8_AIR":
    X_in = {"CHF3":0.2, "C2F6":0.15, "C4F8":0.05, "O2":0.25, "N2":0.35}
else:
    raise RuntimeError("unknown CASE")

track_default = ["CF4","O2","N2","F","CF3","CF2","CO","CO2","COF2","SiF4","CHF3","C2F6","C4F8","O","NO"]
# ===========================

def build_solution(path, drop_electron):
    gas = ct.Solution(path)
    if not drop_electron:
        return gas
    ban = {"E","e-","PHOTON"}
    keep_rxn = []
    for r in gas.reactions():
        names = set(r.reactants.keys()) | set(r.products.keys())
        if any(n in ban for n in names):
            continue
        keep_rxn.append(r)
    species = [gas.species(n) for n in gas.species_names if n not in ban]
    return ct.Solution(thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=keep_rxn)

def make_cstr(gas, X_in, T, P, V, tau):
    gas_in = gas.clone(); gas_in.TPX = T, P, X_in
    inlet = ct.Reservoir(gas_in)

    gas_r = gas.clone(); gas_r.TPX = T, P, X_in
    r = ct.IdealGasReactor(gas_r, energy="off", name="CSTR"); r.volume = V

    outlet = ct.Reservoir(gas_in)

    rho_in = gas_in.density
    mdot = rho_in * V / tau
    mfc = ct.MassFlowController(inlet, r, mdot=mdot)
    pcv = ct.PressureController(r, outlet, master=mfc, K=1e-6)

    sim = ct.ReactorNet([r])
    return sim, r, mdot

def reach_steady(sim, r, t_end):
    last = r.thermo.X.copy()
    while sim.time < t_end:
        sim.advance(min(sim.time + t_end/2000.0, t_end))
        X = r.thermo.X
        if np.max(np.abs(X-last)/np.maximum(1e-20, np.maximum(X,last))) < 1e-9:
            break
        last[:] = X
    return sim.time

def main():
    gas0 = build_solution(MECH, drop_electron=filter_electron_in_yaml)
    print(f"[Loaded] species={gas0.n_species}, reactions={len(gas0.reactions())}")
    if len(gas0.reactions()) < 100:
        print("WARN: reactions < 100. Use a larger mechanism (SAND/Levko).")

    gas0.TPX = T, P, X_in
    sim, R, mdot = make_cstr(gas0, X_in, T, P, V, tau)
    t = reach_steady(sim, R, t_end)

    g = R.thermo
    print("\n=== CSTR steady (approx) ===")
    print(f"T={g.T:.1f} K, P={g.P/ct.one_atm*760.0:.2f} mTorr, time={t:.4f} s, tau={tau:.3f} s")
    print(f"#species={g.n_species}, #reactions={len(g.reactions())}")
    track = [k for k in track_default if k in g.species_names]
    idx = {k: g.species_index(k) for k in track}
    for k,i in idx.items():
        print(f"{k:>8s}: X={g.X[i]:.4e}")

    pairs = sorted([(x, nm) for nm,x in zip(g.species_names, g.X)], reverse=True)[:15]
    print("\n[Top 15 species]")
    for x,nm in pairs:
        print(f"{nm:>12s}: {x:.4e}")

    print("\n[Arrhenius sample (first 12)]")
    for j, rxn in enumerate(g.reactions()[:12], 1):
        rate = getattr(rxn, "rate", None)
        if hasattr(rate,"pre_exponential_factor"):
            print(f"{j:02d}. {rxn.equation:35s}  A={rate.pre_exponential_factor:.3e}  b={rate.temperature_exponent:.2f}  Ea/R={rate.activation_energy/ct.gas_constant:.0f} K")
        else:
            print(f"{j:02d}. {rxn.equation:35s}  (rate type: {type(rate).__name__})")

if __name__ == "__main__":
    main()
