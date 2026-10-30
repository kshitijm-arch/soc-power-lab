"""
power_model.py - Week 1 lab: single-core DVFS energy model.

Everything here is ILLUSTRATIVE and built from public textbook relationships.
No vendor numbers. Replace the constants with your own assumptions and
document them in the README.

Model
-----
P_dyn   = Cdyn * V^2 * f
P_leak  = P_LEAK_REF * (V / V_REF) * exp(GAMMA_V * (V - V_REF)) * exp(BETA_T * (T - T_REF))
P_other = platform / uncore power that is on whenever the core is active
P_active(f) = P_dyn + P_leak + P_other
t(f)    = cycles / f   (compute-bound)   or  cycles_cpu / f + t_mem  (memory-bound)
E(f)    = P_active(f) * t(f)
"""
from dataclasses import dataclass
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters
# V-f table (P-states). Lowest points share Vmin so you can see the
# "voltage cannot drop any further" flattening.
FREQ_GHZ = np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6])
VOLT     = np.array([0.60, 0.60, 0.65, 0.72, 0.80, 0.88, 0.97, 1.07, 1.18])


@dataclass
class Params:
    cdyn_nf: float = 1       # effective switched capacitance (activity x C), nF
    p_leak_ref_w: float = 1.0     # leakage at V_REF and T_REF, watts
    v_ref: float = 1.0
    t_ref_c: float = 60.0
    gamma_v: float = 2.0          # leakage voltage sensitivity (1/V)
    beta_t: float = math.log(2) / 5.0   # leakage doubles about every 25 C
    p_other_w: float = 0.5        # always-on-while-active platform/uncore power
    p_idle_w: float = 0.1         # power when the core is parked in deep idle
    cycles: float = 10e9          # work: 10 billion cycles of core-bound work
    t_mem_s: float = 0.0          # frequency-independent stall time (memory-bound part)
    temp_c: float = 60.0          # fixed junction temp used when thermal feedback is off
    t_amb_c: float = 40.0         # ambient for the optional thermal feedback
    r_th_c_per_w: float = 5.0     # junction-to-ambient thermal resistance


# ------------------------------------------------------------------- physics
def p_dyn(p: Params, v, f_ghz):
    return p.cdyn_nf * 1e-9 * v**2 * f_ghz * 1e9


def p_leak(p: Params, v, temp_c):
    return (p.p_leak_ref_w * (v / p.v_ref)
            * np.exp(p.gamma_v * (v - p.v_ref))
            * np.exp(p.beta_t * (temp_c - p.t_ref_c)))


def solve_temp(p: Params, v, f_ghz, iters=50, t_max=105.0):
    """Optional thermal feedback: T = T_amb + R_th * P(T). Fixed-point iteration.
    Leakage depends on T and T depends on power, so iterate until stable.
    Clamped at t_max so thermal runaway does not blow up the plot."""
    t = p.temp_c
    for _ in range(iters):
        pw = p_dyn(p, v, f_ghz) + p_leak(p, v, t) + p.p_other_w
        t_new = min(p.t_amb_c + p.r_th_c_per_w * pw, t_max)
        if abs(t_new - t) < 1e-4:
            break
        t = t_new
    return t


def sweep(p: Params, thermal_feedback=True):
    """Evaluate every P-state. Returns dict of arrays."""
    f, v = FREQ_GHZ, VOLT
    temp = (np.array([solve_temp(p, vi, fi) for vi, fi in zip(v, f)])
            if thermal_feedback else np.full_like(f, p.temp_c))
    pd, pl = p_dyn(p, v, f), p_leak(p, v, temp)
    pa = pd + pl + p.p_other_w
    t = p.cycles / (f * 1e9) + p.t_mem_s
    return dict(f=f, v=v, temp=temp, p_dyn=pd, p_leak=pl, p_active=pa, t=t, e=pa * t)


def window_energy(res, p: Params, window_s):
    """Race-to-idle accounting: energy over a fixed window.
    Run the task, then sit in idle until the window ends."""
    idle_t = np.maximum(window_s - res["t"], 0.0)
    return res["e"] + p.p_idle_w * idle_t


# ------------------------------------------------------------------ plotting
def mark_min(ax, f, e, label, color):
    i = int(np.argmin(e))
    ax.plot(f[i], e[i], "o", ms=9, mfc="none", mec=color, mew=2)
    ax.annotate(f"{label}: min at {f[i]:.1f} GHz", (f[i], e[i]),
                textcoords="offset points", xytext=(8, 10), color=color, fontsize=8)
    return i


def plot_plot1(p: Params, path):
    r = sweep(p)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    ax = axs[0]
    ax.stackplot(r["f"], r["p_dyn"], r["p_leak"], np.full_like(r["f"], p.p_other_w),
                 labels=["dynamic", "leakage", "other (uncore)"],
                 colors=["#0F7B8A", "#E07A5F", "#B8B8B8"])
    ax.set(xlabel="Frequency (GHz)", ylabel="Active power (W)", title="Power breakdown")
    ax.legend(fontsize=8, loc="upper left")
    axs[1].plot(r["f"], r["t"], "o-", color="#14213D")
    axs[1].set(xlabel="Frequency (GHz)", ylabel="Runtime (s)", title="Runtime per task")
    ax = axs[2]
    ax.plot(r["f"], r["e"], "o-", color="#14213D", label="total (with leakage + other)")
    ax.plot(r["f"], r["p_dyn"] * r["t"], "s--", color="#0F7B8A", label="dynamic only")
    mark_min(ax, r["f"], r["e"], "with leakage", "#14213D")
    ax.set(xlabel="Frequency (GHz)", ylabel="Energy per task (J)",
           title="Energy per task: minimum-energy point")
    ax.legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return r


def plot_plot2(base: Params, path, window_s=15.0):
    low = Params(**{**base.__dict__, "p_other_w": 0.5})
    high = Params(**{**base.__dict__, "p_other_w": 3.0})
    rl, rh = sweep(low), sweep(high)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axs[0]
    ax.plot(rl["f"], rl["e"], "o-", color="#0F7B8A", label="low static (0.5 W)")
    ax.plot(rh["f"], rh["e"], "o-", color="#E07A5F", label="high static (3.0 W)")
    il = mark_min(ax, rl["f"], rl["e"], "low", "#0F7B8A")
    ih = mark_min(ax, rh["f"], rh["e"], "high", "#E07A5F")
    ax.set(xlabel="Frequency (GHz)", ylabel="Energy per task (J)",
           title="Min-energy point moves up with static power")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1]
    deep = Params(**{**high.__dict__, "p_idle_w": 0.1})
    shallow = Params(**{**high.__dict__, "p_idle_w": 2.5})
    ed = window_energy(sweep(deep), deep, window_s)
    es = window_energy(sweep(shallow), shallow, window_s)
    ax.plot(rh["f"], ed, "o-", color="#14213D", label="deep idle (0.1 W) -> race to idle")
    ax.plot(rh["f"], es, "o-", color="#9A8C98", label="shallow idle (2.5 W)")
    mark_min(ax, rh["f"], ed, "deep", "#14213D")
    mark_min(ax, rh["f"], es, "shallow", "#6D6875")
    ax.set(xlabel="Frequency (GHz)", ylabel=f"Energy over {window_s:.0f} s window (J)",
           title="Idle depth decides: race-to-idle vs slow down")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return (rl, il), (rh, ih), (ed, es)


# -------------------------------------------------------------- sanity checks
def sanity_checks(p: Params):
    r = sweep(p)
    # 1. Dynamic power at nominal point matches the hand calculation.
    hand = p.cdyn_nf * 1e-9 * 1.18**2 * 3.6e9
    assert abs(r["p_dyn"][-1] - hand) < 1e-9, "dynamic power mismatch"
    # 2. Leakage rises with temperature.
    assert p_leak(p, 1.0, 80) > p_leak(p, 1.0, 60) > p_leak(p, 1.0, 40)
    # 3. Runtime is exactly cycles / f for a compute-bound task.
    assert np.allclose(r["t"], p.cycles / (r["f"] * 1e9))
    # 4. Interior minimum: neither the slowest nor the fastest state wins.
    i = int(np.argmin(r["e"]))
    assert 0 < i < len(r["f"]) - 1, "minimum is at an endpoint - check parameters"
    # 5. With leakage and uncore removed, the minimum moves to the lowest P-state.
    q = Params(**{**p.__dict__, "p_leak_ref_w": 0.0, "p_other_w": 0.0})
    assert int(np.argmin(sweep(q)["e"])) <= 1, "dynamic-only should favor lowest V/f"
    print("sanity checks passed")


def table(r, p):
    print(f"{'f GHz':>6} {'V':>5} {'T C':>5} {'P_dyn':>7} {'P_leak':>7} {'P_act':>7} {'t s':>7} {'E J':>7}")
    for i in range(len(r["f"])):
        print(f"{r['f'][i]:6.1f} {r['v'][i]:5.2f} {r['temp'][i]:5.0f} {r['p_dyn'][i]:7.2f} "
              f"{r['p_leak'][i]:7.2f} {r['p_active'][i]:7.2f} {r['t'][i]:7.2f} {r['e'][i]:7.2f}")


if __name__ == "__main__":
    p = Params()
    sanity_checks(p)
    r = plot_plot1(p, "plot1_energy_vs_freq.png")
    print("\nPlot 1 sweep (fixed T = 60 C):"); table(r, p)
    i = int(np.argmin(r["e"]))
    print(f"-> minimum-energy point: {r['f'][i]:.1f} GHz / {r['v'][i]:.2f} V, {r['e'][i]:.2f} J")
    # Stretch: thermal feedback
    rt = sweep(p, thermal_feedback=True)
    it = int(np.argmin(rt["e"]))
    print(f"-> with thermal feedback: min at {rt['f'][it]:.1f} GHz (Tj at 3.6 GHz = {rt['temp'][-1]:.0f} C)")

    (rl, il), (rh, ih), (ed, es) = plot_plot2(p, "plot2_race_to_idle.png")
    print(f"\nLow static : min-energy {rl['f'][il]:.1f} GHz")
    print(f"High static: min-energy {rh['f'][ih]:.1f} GHz")
    print(f"Window, deep idle   : best {rh['f'][int(np.argmin(ed))]:.1f} GHz")
    print(f"Window, shallow idle: best {rh['f'][int(np.argmin(es))]:.1f} GHz")
