# example_h2o2_constP.py
# pip install cantera matplotlib
import csv

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# --- 機構（29反応） ---
gas = ct.Solution("h2o2.yaml")

# 初期条件（等量比 φ=1 の H2/O2、N2 で少し希釈）
T0 = 1000.0            # K
P0 = ct.one_atm        # Pa
X0 = {"H2": 2.0, "O2": 1.0, "N2": 7.52}  # 空気希釈っぽく

gas.TPX = T0, P0, X0

# 定圧・断熱の 0次元バッチ反応器
r = ct.IdealGasConstPressureReactor(gas, energy="on")
sim = ct.ReactorNet([r])

species_names = gas.species_names

t_end = 5e-3    # s
dt    = 1e-6    # s

ts = []
profiles = []
Ts = []

# 時間発展
t = 0.0
while t < t_end:
    t += dt
    sim.advance(t)
    ts.append(sim.time)
    Ts.append(r.T)
    profiles.append(r.thermo.X.copy())

ts = np.array(ts)
profiles = np.array(profiles)
Ts = np.array(Ts)

# 結果表示
print(f"Final time: {ts[-1]*1e3:.3f} ms")
print(f"Final T:    {Ts[-1]:.1f} K")

# 最終組成（上位8種）を表示
final_X = profiles[-1]
top_indices = np.argsort(final_X)[::-1][:8]
print("Final mole fractions (top 8 species):")
for i in top_indices:
    print(f"  {species_names[i]:5s}: {final_X[i]:.5f}")

# Arrhenius 係数も確認（先頭いくつかを出力）
print("\nSome Arrhenius parameters from the mechanism (first 12 reactions):")
for i, rxn in enumerate(gas.reactions()[:12], start=1):
    rate = getattr(rxn, "rate", None)
    if rate and hasattr(rate, "pre_exponential_factor"):
        A = rate.pre_exponential_factor
        b = rate.temperature_exponent
        # Ea は J/kmol を返す場合があるので、Ea/R[K]で表示すると安全
        Ta = rate.activation_energy / ct.gas_constant
        print(f"{i:02d}. {rxn.equation:30s}  A={A:.3e}, b={b:.2f}, Ea/R={Ta:.0f} K")
    else:
        print(f"{i:02d}. {rxn.equation:30s}  (rate type: {type(rate).__name__})")

# 簡単なプロット（任意）
plt.figure(figsize=(10, 6))
time_ms = ts * 1e3
epsilon = 1e-30  # 対数表示のための下限値
for i, name in enumerate(species_names):
    plt.plot(time_ms, np.clip(profiles[:, i], epsilon, None), label=name)
plt.xlabel("time [ms]")
plt.ylabel("mole fraction [-]")
plt.yscale("log")
plt.legend(fontsize=7, ncol=2, loc="upper right")
plt.title("H2/O2 ignition at 1 atm, const-P adiabatic (h2o2.yaml)")
plt.tight_layout()
plt.show()

# CSV 出力
output_csv = "composition_profiles.csv"
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time_s"] + list(species_names))
    for time_val, mole_fractions in zip(ts, profiles):
        writer.writerow([time_val] + mole_fractions.tolist())

print(f"\nSaved mole fraction history to {output_csv}")
