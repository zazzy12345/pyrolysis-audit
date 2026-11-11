from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
PARAM_PATH = ROOT.parent / "public_tga/sources/PMMA/Material_Properties/2021/MaCFP_PMMA_NIST.json"



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        next(reader)  # units
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                try:
                    x, y = float(row[0]), float(row[1])
                    if math.isfinite(x) and math.isfinite(y):
                        xs.append(x)
                        ys.append(y)
                except ValueError:
                    pass
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def load_temp_csv(path: Path) -> dict:
    """Load back-surface temperature CSV with explicit valid-channel masking."""
    ts, mean_temps = [], []
    ch_data: dict[str, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        next(reader)  # units
        ch_names = [h.strip() for h in hdr[1:]]
        for cn in ch_names:
            ch_data[cn] = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                t = float(row[0])
            except ValueError:
                continue
            vals = []
            for idx, cell in enumerate(row[1:]):
                if idx < len(ch_names) and cell.strip():
                    try:
                        v = float(cell)
                        if math.isfinite(v):
                            vals.append(v)
                            ch_data[ch_names[idx]].append((t, v))
                    except ValueError:
                        pass
            if vals:
                ts.append(t)
                mean_temps.append(float(np.mean(vals)))
    return {
        "t_s": np.array(ts, dtype=float),
        "T_mean_K": np.array(mean_temps, dtype=float),
        "channels": {
            cn: {
                "count": len(pts),
                "t_min_s": pts[0][0] if pts else None,
                "t_max_s": pts[-1][0] if pts else None,
                "T_start_K": pts[0][1] if pts else None,
                "T_end_K": pts[-1][1] if pts else None,
            }
            for cn, pts in ch_data.items()
        },
    }


def kaowool_k(T: np.ndarray) -> np.ndarray:
    # Table 4 of NIST Gasification Apparatus README:
    # 260 C (533.15 K) -> 0.0576; 538 C (811.15 K) -> 0.085; 816 C (1089.15 K) -> 0.125; 1093 C (1366.15 K) -> 0.183
    T_pts = np.array([533.15, 811.15, 1089.15, 1366.15], dtype=float)
    k_pts = np.array([0.0576, 0.0850, 0.1250, 0.1830], dtype=float)
    return np.interp(T, T_pts, k_pts)


def q_star_factor(t: float) -> float:
    # Table 1 of NIST Gasification Apparatus README (50 kW/m2 shutter warmup factor)
    t_knots = np.array([0.0, 5.5, 35.0, 90.0, 150.0, 210.0, 270.0, 450.0, 1000.0])
    q_knots = np.array([0.9300, 0.9401, 0.9598, 0.9831, 0.9946, 0.9987, 1.0000, 1.0004, 1.0004])
    return float(np.interp(t, t_knots, q_knots))


def c90_1d(source_amount: np.ndarray, cap: np.ndarray, q: float = 0.90) -> float:
    s = source_amount.ravel()
    m = cap.ravel()
    tot = float(s.sum())
    if tot <= 0.0:
        return 0.0
    order = np.argsort(s / np.maximum(m, 1e-30))[::-1]
    s, m = s[order], m[order]
    target = q * tot
    cum = np.cumsum(s)
    hit = int(np.searchsorted(cum, target, side="left"))
    prev = 0.0 if hit == 0 else float(cum[hit - 1])
    frac = (target - prev) / float(s[hit])
    return float((m[:hit].sum() + frac * m[hit]) / m.sum())


def d90_1d(source_amount: np.ndarray, dy: np.ndarray, q: float = 0.90) -> float:
    s = source_amount.ravel()
    tot = float(s.sum())
    if tot <= 0.0:
        return 0.0
    target = q * tot
    cum = np.cumsum(s)
    hit = int(np.searchsorted(cum, target, side="left"))
    prev = 0.0 if hit == 0 else float(cum[hit - 1])
    frac = (target - prev) / float(s[hit])
    return float(dy[:hit].sum() + frac * dy[hit])


def solve_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    ab = np.empty((3, len(d)), dtype=float)
    ab[0, 0] = 0.0
    ab[0, 1:] = c[:-1]
    ab[1, :] = b
    ab[2, :-1] = a[1:]
    ab[2, -1] = 0.0
    return solve_banded((1, 1), ab, d, overwrite_ab=True, overwrite_b=False, check_finite=False)


def run_1d_slab(
    mode: str,
    L_pmma: float = 0.0059,
    N_pmma: int = 120,
    first_dy: float = 10.0e-6,
    L_ins: float = 0.02852,
    N_ins: int = 60,
    t_end: float = 400.0,
    dt: float = 0.01,
) -> dict:
    """
    1D MaCFP PMMA slab on Kaowool PM insulation under 50 kW/m2 NIST Gasification Apparatus radiant heating.
    All PMMA properties are read directly from MaCFP_PMMA_NIST.json (commit 579b88265a329878a68e1161a8b9ceb5e62a9979):
      A = 2.85e13 s^-1, E = 191000 J/mol, n = 1, nu_s = 0 (K_e = 1),
      rho_0 = 1100 kg/m3, c_p = 2200 J/(kg K), kappa = 0.20 W/(m K),
      Delta_h_py = 8.70e5 J/kg, emissivity = 0.90, surface absorption (Absorption = 'inf').
    Kaowool PM backing uses total thickness L_ins = 28.52 mm (5.72 mm bonded + 22.80 mm stack) and Table 4 properties.
    """
    params = json.loads(PARAM_PATH.read_text(encoding="utf-8"))
    A = float(params["Kinetics"]["Pre-exponential"])       # 2.85e13 s^-1
    E = float(params["Kinetics"]["Activation Energy"])     # 1.91e5 J/mol
    R_u = 8.31446261815324
    rho_p = float(params["Thermodynamics"]["Density"]["Value"])         # 1100 kg/m3
    cp_p = float(params["Thermodynamics"]["Heat Capacity"]["Value"])    # 2200 J/(kg K)
    k_p = float(params["Transport"]["Conductivity"]["Value"])           # 0.20 W/(m K)
    dh_py = float(params["Thermodynamics"]["Heat of Pyrolysis"]["Value"])  # 8.7e5 J/kg
    eps = float(params["Transport"]["Emissivity"]["Value"])             # 0.90
    sigma = 5.670374419e-8

    # NIST Gasification Apparatus boundary conditions (README.md)
    q_steady = 49300.0   # W/m2 average measured over sample area r < 3.5 cm for 50 kW/m2 target
    T_init = 292.6       # K (measured initial back-surface temperature in R3-R5)
    T_wall = 289.15      # K (16 C water-cooled chamber wall)
    T_gas = 298.15       # K (25 C nitrogen flow)
    h_conv = 10.0        # W/(m2 K)

    # Geometric stretched grid in PMMA
    lo, hi = 1.0, 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if first_dy * (mid**N_pmma - 1.0) / (mid - 1.0) > L_pmma:
            hi = mid
        else:
            lo = mid
    r_p = 0.5 * (lo + hi)
    dy_p0 = np.array([first_dy * (r_p**i) for i in range(N_pmma)], dtype=float)
    dy_p0[-1] = L_pmma - float(np.sum(dy_p0[:-1]))
    M_p0 = rho_p * dy_p0  # kg/m2 per cell (reference capacity in material coordinates)

    # Stretched grid in Kaowool PM insulation
    lo, hi = 1.0, 1.5
    first_ins = 80.0e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if first_ins * (mid**N_ins - 1.0) / (mid - 1.0) > L_ins:
            hi = mid
        else:
            lo = mid
    r_i = 0.5 * (lo + hi)
    dy_ins = np.array([first_ins * (r_i**i) for i in range(N_ins)], dtype=float)
    dy_ins[-1] = L_ins - float(np.sum(dy_ins[:-1]))
    rho_ins = 256.0
    cp_ins = 1070.0
    M_ins = rho_ins * dy_ins

    N_tot = N_pmma + N_ins
    T = np.full(N_tot, T_init, dtype=float)
    D = np.zeros(N_pmma, dtype=float)
    chi_eul = np.ones(N_pmma, dtype=float)

    n_steps = int(round(t_end / dt))
    save_every = max(1, int(round(0.1 / dt)))  # save every 0.1 s
    n_save = n_steps // save_every + 1

    t_hist = np.zeros(n_save)
    j0_hist = np.zeros(n_save)
    jchi_hist = np.zeros(n_save)
    es_hist = np.zeros(n_save)
    eb_hist = np.zeros(n_save)
    cov_hist = np.zeros(n_save)
    bs_hist = np.zeros(n_save)
    jensen_hist = np.zeros(n_save)
    c90_pot_hist = np.zeros(n_save)
    c90_act_hist = np.zeros(n_save)
    d90_pot_hist = np.zeros(n_save)
    d90_act_hist = np.zeros(n_save)
    t_surf_hist = np.zeros(n_save)
    t_back_hist = np.zeros(n_save)
    t_back_arith_hist = np.zeros(n_save)
    mass_frac_hist = np.zeros(n_save)
    max_loss_hist = np.zeros(n_save)

    cum_py_heat_J_m2 = 0.0
    snap_4647 = {}

    crossings = {}
    tols = (0.05, 0.10, 0.20)
    for tol in tols:
        crossings[f"{int(tol*100)}pct"] = {
            "Bs": None,
            "Jensen": None,
            "Max_cell_loss_nominal": None,
            "Max_cell_loss_consistent_eps_over_1_plus_eps": None,
            "Eb": None,
        }

    profiles_at_times = {}
    y_centers_0 = np.cumsum(dy_p0) - 0.5 * dy_p0

    for step in range(n_steps + 1):
        t = step * dt
        T_p = T[:N_pmma]
        k_arr = A * np.exp(-E / (R_u * T_p))
        chi_pas = np.exp(-D)
        loss_pas = 1.0 - chi_pas

        pot_cell = M_p0 * k_arr
        act_cell = pot_cell * chi_pas
        j0 = float(np.sum(pot_cell))
        jchi = float(np.sum(act_cell))
        if j0 > 1e-30:
            w = pot_cell / j0
            es = float(np.sum(w * loss_pas))
            deff = float(np.sum(w * D))
        else:
            w = M_p0 / float(np.sum(M_p0))
            es = 0.0
            deff = 0.0
        eb = float(np.sum(M_p0 * loss_pas) / np.sum(M_p0))
        cov_term = es - eb
        bs = es / max(1.0 - es, 1e-15)
        jensen = math.exp(min(deff, 50.0)) - 1.0
        max_loss = float(np.max(loss_pas))

        # Effective last PMMA cell thickness and first insulation cell resistance for interface temperature T_back
        dy_last_p = max(float(dy_p0[-1] * chi_pas[-1]), 1e-9) if mode == "finite_compacted" else float(dy_p0[-1])
        R_p_int = 0.5 * dy_last_p / k_p
        k_ins_0 = float(kaowool_k(np.array([T[N_pmma]]))[0])
        R_ins_int = 0.5 * float(dy_ins[0]) / k_ins_0
        t_back_res = float((R_ins_int * T[N_pmma - 1] + R_p_int * T[N_pmma]) / (R_p_int + R_ins_int))
        t_back_arith = float(0.5 * (T[N_pmma - 1] + T[N_pmma]))

        if abs(t - 46.47) < 0.5 * dt:
            snap_4647 = {
                "time_s": round(t, 4),
                "T_surf_K": round(float(T[0]), 4),
                "T_back_K": round(t_back_res, 4),
                "J0_g_m2_s": round(1000.0 * j0, 5),
                "Jchi_g_m2_s": round(1000.0 * jchi, 5),
                "Es_pct": round(100.0 * es, 5),
                "Eb_pct": round(100.0 * eb, 6),
                "Bs_pct": round(100.0 * bs, 5),
                "Jensen_pct": round(100.0 * jensen, 5),
                "d90_pot_um": round(1e6 * d90_1d(pot_cell, dy_p0), 3),
                "C90_pot_pct": round(100.0 * c90_1d(pot_cell, M_p0), 4),
            }

        def _make_snap():
            return {
                "time_s": round(t, 4),
                "T_surf_K": round(float(T[0]), 3),
                "J0_g_m2_s": round(1000.0 * j0, 5),
                "Jchi_g_m2_s": round(1000.0 * jchi, 5),
                "Es_pct": round(100 * es, 4),
                "Cov_pct": round(100 * cov_term, 4),
                "Bs_pct": round(100 * bs, 4),
                "Jensen_pct": round(100 * jensen, 4),
                "Eb_pct": round(100 * eb, 4),
                "max_loss_pct": round(100 * max_loss, 4),
                "d90_pot_um": round(1e6 * d90_1d(pot_cell, dy_p0), 2),
                "C90_pot_pct": round(100 * c90_1d(pot_cell, M_p0), 4),
            }

        for tol in tols:
            tag = f"{int(tol*100)}pct"
            loss_cons = tol / (1.0 + tol)
            if crossings[tag]["Jensen"] is None and jensen > tol:
                crossings[tag]["Jensen"] = _make_snap()
            if crossings[tag]["Bs"] is None and bs > tol:
                crossings[tag]["Bs"] = _make_snap()
            if crossings[tag]["Max_cell_loss_nominal"] is None and max_loss > tol:
                crossings[tag]["Max_cell_loss_nominal"] = _make_snap()
            if crossings[tag]["Max_cell_loss_consistent_eps_over_1_plus_eps"] is None and max_loss > loss_cons:
                crossings[tag]["Max_cell_loss_consistent_eps_over_1_plus_eps"] = _make_snap()
            if crossings[tag]["Eb"] is None and eb > tol:
                crossings[tag]["Eb"] = _make_snap()

        if step % save_every == 0:
            idx = step // save_every
            t_hist[idx] = t
            j0_hist[idx] = 1000.0 * j0      # g/(m2 s)
            jchi_hist[idx] = 1000.0 * jchi  # g/(m2 s)
            es_hist[idx] = 100.0 * es
            eb_hist[idx] = 100.0 * eb
            cov_hist[idx] = 100.0 * cov_term
            bs_hist[idx] = 100.0 * bs
            jensen_hist[idx] = 100.0 * jensen
            c90_pot_hist[idx] = 100.0 * c90_1d(pot_cell, M_p0)
            c90_act_hist[idx] = 100.0 * c90_1d(act_cell, M_p0)
            d90_pot_hist[idx] = 1e6 * d90_1d(pot_cell, dy_p0)
            d90_act_hist[idx] = 1e6 * d90_1d(act_cell, dy_p0)
            t_surf_hist[idx] = float(T[0])
            t_back_hist[idx] = t_back_res
            t_back_arith_hist[idx] = t_back_arith
            mass_frac_hist[idx] = 1.0 - eb
            max_loss_hist[idx] = 100.0 * max_loss

        if int(round(t * 100)) in (2000, 2500, 3000, 4000, 6000):
            profiles_at_times[f"{t:.1f}s"] = {
                "y_um": (1e6 * y_centers_0).tolist(),
                "T_K": T_p.tolist(),
                "chi_pas": chi_pas.tolist(),
                "k_per_s": k_arr.tolist(),
            }

        if step == n_steps:
            break

        # Advance 1 step via linearized implicit backward Euler in 1D finite-volume coordinates
        if mode == "undepleted":
            chi_active = np.ones(N_pmma, dtype=float)
            dy_p_eff = dy_p0
            M_p_eff = M_p0
        elif mode == "finite_fixed":
            chi_active = chi_pas
            dy_p_eff = dy_p0
            M_p_eff = M_p0
        elif mode == "finite_compacted":
            chi_active = chi_pas
            dy_p_eff = np.maximum(dy_p0 * chi_pas, 1e-9)
            M_p_eff = np.maximum(M_p0 * chi_pas, 1e-6)
        else:
            raise ValueError(mode)

        dy_all = np.concatenate([dy_p_eff, dy_ins])
        k_ins_vals = kaowool_k(T[N_pmma:])
        k_cond = np.concatenate([np.full(N_pmma, k_p), k_ins_vals])
        C_cell = np.concatenate([M_p_eff * cp_p, M_ins * cp_ins])

        G = 1.0 / (0.5 * dy_all[:-1] / k_cond[:-1] + 0.5 * dy_all[1:] / k_cond[1:])

        Ts = T[0]
        q_inc = q_steady * q_star_factor(t)
        q_rad_0 = eps * q_inc - eps * sigma * (Ts**4 - T_wall**4) - h_conv * (Ts - T_gas)
        dq_dTs = -4.0 * eps * sigma * (Ts**3) - h_conv

        sink = np.zeros(N_tot, dtype=float)
        dsink_dT = np.zeros(N_tot, dtype=float)
        r_py = M_p0 * k_arr * chi_active
        sink[:N_pmma] = dh_py * r_py
        dsink_dT[:N_pmma] = dh_py * r_py * (E / (R_u * (T_p**2)))

        a = np.zeros(N_tot, dtype=float)
        b = C_cell / dt + dsink_dT
        c = np.zeros(N_tot, dtype=float)
        rhs = (C_cell / dt) * T - sink + dsink_dT * T

        b[0] -= dq_dTs
        rhs[0] += q_rad_0 - dq_dTs * Ts

        b[:-1] += G
        b[1:] += G
        a[1:] = -G
        c[:-1] = -G

        T_new = solve_tridiag(a, b, c, rhs)
        sink_impl_p = sink[:N_pmma] + dsink_dT[:N_pmma] * (T_new[:N_pmma] - T[:N_pmma])
        cum_py_heat_J_m2 += float(np.sum(sink_impl_p)) * dt
        k_new = A * np.exp(-E / (R_u * T_new[:N_pmma]))
        D += 0.5 * dt * (k_arr + k_new)
        chi_eul *= np.maximum(0.0, 1.0 - dt * k_arr)
        T = T_new

    m_act_consumed = float(np.sum(M_p0 * (1.0 - np.exp(-D))))
    m_heat_equiv = cum_py_heat_J_m2 / dh_py
    R_p_0 = 0.5 * float(dy_p0[-1]) / k_p
    R_ins_0 = 0.5 * float(dy_ins[0]) / float(kaowool_k(np.array([T_init]))[0])
    w_p_0 = R_ins_0 / (R_p_0 + R_ins_0)
    w_ins_0 = R_p_0 / (R_p_0 + R_ins_0)

    return {
        "mode": mode,
        "L_pmma_mm": 1e3 * L_pmma,
        "L_ins_mm": 1e3 * L_ins,
        "N_pmma": N_pmma,
        "first_dy_um": 1e6 * first_dy,
        "r_p": float(r_p),
        "r_ins": float(r_i),
        "w_p_0": float(w_p_0),
        "w_ins_0": float(w_ins_0),
        "dt_s": dt,
        "t_s": t_hist,
        "J0_g_m2_s": j0_hist,
        "Jchi_g_m2_s": jchi_hist,
        "E_source_pct": es_hist,
        "E_bulk_pct": eb_hist,
        "cov_term_pct": cov_hist,
        "B_s_pct": bs_hist,
        "Jensen_pct": jensen_hist,
        "C90_pot_pct": c90_pot_hist,
        "C90_act_pct": c90_act_hist,
        "d90_pot_um": d90_pot_hist,
        "d90_act_um": d90_act_hist,
        "T_surf_K": t_surf_hist,
        "T_back_K": t_back_hist,
        "T_back_arith_K": t_back_arith_hist,
        "mass_frac": mass_frac_hist,
        "max_loss_pct": max_loss_hist,
        "crossings": crossings,
        "snap_4647": snap_4647,
        "sink_activity_consistency": {
            "activity_consumed_mass_kg_m2": round(m_act_consumed, 6),
            "heat_sink_equivalent_mass_kg_m2": round(m_heat_equiv, 6),
            "rel_diff_pct": round(100.0 * (m_heat_equiv - m_act_consumed) / max(m_act_consumed, 1e-15), 4),
            "pyrolysis_energy_diff_kJ_m2": round(1e-3 * (cum_py_heat_J_m2 - dh_py * m_act_consumed), 3),
            "complete_initial_mass_pyrolysis_heat_MJ_m2": round(1e-6 * dh_py * float(np.sum(M_p0)), 5),
        },
        "profiles": profiles_at_times,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running 1D MaCFP PMMA slab models (L_ins = 28.52 mm total backing, t_end = 450 s)...")
    res_undep = run_1d_slab("undepleted", L_pmma=0.0059, L_ins=0.02852, t_end=450.0, dt=0.01)
    res_fixed = run_1d_slab("finite_fixed", L_pmma=0.0059, L_ins=0.02852, t_end=450.0, dt=0.01)
    res_comp = run_1d_slab("finite_compacted", L_pmma=0.0059, L_ins=0.02852, t_end=450.0, dt=0.01)

    # Check backing-thickness sensitivity (22.80 mm vs 28.52 mm)
    res_comp_228 = run_1d_slab("finite_compacted", L_pmma=0.0059, L_ins=0.02280, t_end=450.0, dt=0.01)
    max_diff_mlr_backing = float(np.max(np.abs(res_comp["Jchi_g_m2_s"][:3501] - res_comp_228["Jchi_g_m2_s"][:3501])))
    max_diff_tback_backing = float(np.max(np.abs(res_comp["T_back_K"][:3501] - res_comp_228["T_back_K"][:3501])))
    print(f"Backing sensitivity (28.52 vs 22.80 mm over 0-350s): max |dMLR|={max_diff_mlr_backing:.3e} g/m2/s, max |dT_back|={max_diff_tback_backing:.3e} K")

    # 1D mesh & time-step refinement study (Major Comment 3.7 & Round 3 Section 4.3)
    t_mlr_r3, mlr_r3 = load_csv_xy(DATA_DIR / "MaCFP-PMMA_Gasification_q50_MLR_R3.csv")
    mask_r3_350 = (t_mlr_r3 >= 0) & (t_mlr_r3 <= 350)
    grid_study = {}
    grid_curves = {}
    for label, n_p, dy1, dt_val in (
        ("coarse_60_dy20um_dt20ms", 60, 20.0e-6, 0.02),
        ("baseline_120_dy10um_dt10ms", 120, 10.0e-6, 0.01),
        ("fine_240_dy5um_dt5ms", 240, 5.0e-6, 0.005),
    ):
        ru = run_1d_slab("undepleted", L_pmma=0.0059, N_pmma=n_p, first_dy=dy1, L_ins=0.02852, t_end=100.0, dt=dt_val)
        rc = run_1d_slab("finite_compacted", L_pmma=0.0059, N_pmma=n_p, first_dy=dy1, L_ins=0.02852, t_end=420.0, dt=dt_val)
        i_465 = int(round(46.5 / 0.1))
        i_600 = int(round(60.0 / 0.1))
        i_350 = int(round(350.0 / 0.1))
        i_pk = int(np.argmax(rc["Jchi_g_m2_s"]))
        mod_r3 = np.interp(t_mlr_r3[mask_r3_350], rc["t_s"], rc["Jchi_g_m2_s"])
        rmse_r3 = float(np.sqrt(np.mean((mod_r3 - mlr_r3[mask_r3_350]) ** 2)))
        grid_study[label] = {
            "N_pmma": n_p,
            "first_dy_um": 1e6 * dy1,
            "dt_ms": 1e3 * dt_val,
            "5pct_Jensen_crossing_s": ru["crossings"]["5pct"]["Jensen"]["time_s"],
            "5pct_Bs_crossing_s": ru["crossings"]["5pct"]["Bs"]["time_s"],
            "d90_at_5pct_crossing_um": ru["crossings"]["5pct"]["Bs"]["d90_pot_um"],
            "d90_at_60s_um": float(ru["d90_pot_um"][i_600]),
            "Bs_at_60s_pct": float(ru["B_s_pct"][i_600]),
            "MLR_compacted_46p5s_g_m2_s": round(float(rc["Jchi_g_m2_s"][i_465]), 3),
            "MLR_compacted_60s_g_m2_s": round(float(rc["Jchi_g_m2_s"][i_600]), 3),
            "MLR_compacted_350s_g_m2_s": round(float(rc["Jchi_g_m2_s"][i_350]), 3),
            "MLR_compacted_full_peak_g_m2_s": round(float(rc["Jchi_g_m2_s"][i_pk]), 3),
            "MLR_compacted_full_peak_time_s": round(float(rc["t_s"][i_pk]), 2),
            "MLR_RMSE_0_350s_R3_g_m2_s": round(rmse_r3, 3),
        }
        grid_curves[label] = {
            "t_100_s": ru["t_s"][:1001].tolist(),
            "Bs_100_pct": ru["B_s_pct"][:1001].tolist(),
            "Jensen_100_pct": ru["Jensen_pct"][:1001].tolist(),
            "t_420_s": rc["t_s"][::10].tolist(),
            "Jchi_compacted_420_g_m2_s": rc["Jchi_g_m2_s"][::10].tolist(),
        }
    print("1D mesh/dt sensitivity study:", json.dumps(grid_study, indent=2))

    # Table 1 q*(t) interval-average verification (Round 3 Section 3)
    table1_intervals = [
        ("1-10 s", 1.0, 10.0, 0.9401),
        ("10-60 s", 10.0, 60.0, 0.9598),
        ("60-120 s", 60.0, 120.0, 0.9831),
        ("120-180 s", 120.0, 180.0, 0.9946),
        ("180-240 s", 180.0, 240.0, 0.9987),
        ("240-300 s", 240.0, 300.0, 1.0000),
        ("300-600 s", 300.0, 600.0, 1.0004),
    ]
    q_star_verification = []
    for label_int, t_a, t_b, q_exp in table1_intervals:
        ts_fine = np.linspace(t_a, t_b, 1001)
        qs_fine = np.array([q_star_factor( float(tt) ) for tt in ts_fine])
        q_avg = float(np.trapezoid(qs_fine, ts_fine) / (t_b - t_a))
        q_star_verification.append({
            "interval": label_int,
            "README_Table1_mean": q_exp,
            "executed_piecewise_linear_mean": round(q_avg, 4),
            "abs_diff": round(abs(q_avg - q_exp), 4),
        })
    print("Table 1 q*(t) interval verification:", json.dumps(q_star_verification, indent=2))

    # Full burnout peak vs 0-350 s maximum
    idx_350 = int(round(350.0 / 0.1))
    idx_peak_full = int(np.argmax(res_comp["Jchi_g_m2_s"]))
    print(
        f"Compacted 5.90 mm model: max(0-350s)={res_comp['Jchi_g_m2_s'][:idx_350+1].max():.2f} g/m2/s at {res_comp['t_s'][np.argmax(res_comp['Jchi_g_m2_s'][:idx_350+1])]:.1f}s; "
        f"full-burnout peak={res_comp['Jchi_g_m2_s'][idx_peak_full]:.2f} g/m2/s at {res_comp['t_s'][idx_peak_full]:.1f}s"
    )

    # Compare both representative 5.90 mm slab and specimen-specific thicknesses for R3, R4, R5 and Temp_R1..R5
    exp_metrics = {}
    for r_id, L_mm, m0_g in (("R3", 5.90, 26.689), ("R4", 5.65, 25.643), ("R5", 6.20, 28.113)):
        t_mlr, mlr = load_csv_xy(DATA_DIR / f"MaCFP-PMMA_Gasification_q50_MLR_{r_id}.csv")
        t_m, m_g = load_csv_xy(DATA_DIR / f"MaCFP-PMMA_Gasification_q50_Mass_{r_id}.csv")
        m_frac = m_g / m0_g
        rc_spec = res_comp if abs(L_mm - 5.90) < 1e-6 else run_1d_slab("finite_compacted", L_pmma=L_mm * 1e-3, L_ins=0.02852, t_end=450.0, dt=0.01)
        mask = (t_mlr >= 0) & (t_mlr <= 350)
        mod_mlr_rep = np.interp(t_mlr[mask], res_comp["t_s"], res_comp["Jchi_g_m2_s"])
        mod_mlr_spec = np.interp(t_mlr[mask], rc_spec["t_s"], rc_spec["Jchi_g_m2_s"])
        mask_m = (t_m >= 0) & (t_m <= 350) & (m_frac >= 0.05)
        mod_m_rep = np.interp(t_m[mask_m], res_comp["t_s"], res_comp["mass_frac"])
        mod_m_spec = np.interp(t_m[mask_m], rc_spec["t_s"], rc_spec["mass_frac"])
        i_pk_exp = int(np.argmax(mlr))
        i_pk_spec = int(np.argmax(rc_spec["Jchi_g_m2_s"]))
        exp_metrics[f"Mass_MLR_{r_id}"] = {
            "L_mm": L_mm,
            "initial_mass_g": m0_g,
            "measured_peak_MLR_g_m2_s": round(float(mlr[i_pk_exp]), 2),
            "measured_peak_time_s": round(float(t_mlr[i_pk_exp]), 1),
            "measured_MLR_at_46p47s_g_m2_s": round(float(np.interp(46.47, t_mlr, mlr)), 2),
            "measured_MLR_at_60s_g_m2_s": round(float(np.interp(60.00, t_mlr, mlr)), 2),
            "measured_onset_2g_m2_s_time_s": round(float(t_mlr[np.flatnonzero(mlr >= 2.0)[0]]), 1),
            "model_rep_5p90mm_max_0_350s_g_m2_s": round(float(res_comp["Jchi_g_m2_s"][:idx_350+1].max()), 2),
            "model_spec_L_max_0_350s_g_m2_s": round(float(rc_spec["Jchi_g_m2_s"][:idx_350+1].max()), 2),
            "model_spec_full_peak_MLR_g_m2_s": round(float(rc_spec["Jchi_g_m2_s"][i_pk_spec]), 2),
            "model_spec_full_peak_time_s": round(float(rc_spec["t_s"][i_pk_spec]), 1),
            "model_onset_2g_m2_s_time_s": round(float(rc_spec["t_s"][np.flatnonzero(rc_spec["Jchi_g_m2_s"] >= 2.0)[0]]), 1),
            "MLR_RMSE_0_350s_rep_5p90mm_g_m2_s": round(float(np.sqrt(np.mean((mod_mlr_rep - mlr[mask]) ** 2))), 2),
            "MLR_RMSE_0_350s_specimen_L_g_m2_s": round(float(np.sqrt(np.mean((mod_mlr_spec - mlr[mask]) ** 2))), 2),
            "mass_frac_RMSE_rep_5p90mm_pct": round(100.0 * float(np.sqrt(np.mean((mod_m_rep - m_frac[mask_m]) ** 2))), 2),
            "mass_frac_RMSE_specimen_L_pct": round(100.0 * float(np.sqrt(np.mean((mod_m_spec - m_frac[mask_m]) ** 2))), 2),
        }

    for r_id, L_mm in (("R1", 5.35), ("R2", 5.52), ("R3", 6.12), ("R4", 6.10), ("R5", 6.25)):
        td = load_temp_csv(DATA_DIR / f"MaCFP-PMMA_Gasification_q50_Temp_{r_id}.csv")
        t_tb, tb = td["t_s"], td["T_mean_K"]
        rc_spec = run_1d_slab("finite_compacted", L_pmma=L_mm * 1e-3, L_ins=0.02852, t_end=250.0, dt=0.01)
        mod_tb_rep = np.interp(t_tb, res_comp["t_s"], res_comp["T_back_K"])
        mod_tb_spec = np.interp(t_tb, rc_spec["t_s"], rc_spec["T_back_K"])
        mod_tb_rep_arith = np.interp(t_tb, res_comp["t_s"], res_comp["T_back_arith_K"])
        mod_tb_spec_arith = np.interp(t_tb, rc_spec["t_s"], rc_spec["T_back_arith_K"])
        exp_metrics[f"Temp_{r_id}"] = {
            "specimen_L_mm": L_mm,
            "r_p_specimen": round(rc_spec["r_p"], 12),
            "archived_time_range_s": [float(t_tb.min()), float(t_tb.max())],
            "channels": td["channels"],
            "T_back_RMSE_rep_5p90mm_K": round(float(np.sqrt(np.mean((mod_tb_rep - tb) ** 2))), 2),
            "T_back_RMSE_specimen_L_K": round(float(np.sqrt(np.mean((mod_tb_spec - tb) ** 2))), 2),
            "T_back_arith_RMSE_rep_5p90mm_K": round(float(np.sqrt(np.mean((mod_tb_rep_arith - tb) ** 2))), 2),
            "T_back_arith_RMSE_specimen_L_K": round(float(np.sqrt(np.mean((mod_tb_spec_arith - tb) ** 2))), 2),
            "measured_T_end_K": round(float(tb[-1]), 2),
            "model_spec_T_end_K": round(float(mod_tb_spec[-1]), 2),
        }
    max_tback_res_vs_arith_210s = float(np.max(np.abs(res_comp["T_back_K"][:2101] - res_comp["T_back_arith_K"][:2101])))
    print(f"Mesh ratios: r_p(5.90mm)={res_comp['r_p']:.12f}, r_ins(28.52mm)={res_comp['r_ins']:.12f}, w_p_0={res_comp['w_p_0']:.4f}, w_ins_0={res_comp['w_ins_0']:.4f}")
    print(f"Max |T_back_res - T_back_arith| over 0-210s = {max_tback_res_vs_arith_210s:.4f} K")
    print("Exact 46.47s undep snapshot:", res_undep["snap_4647"])
    print("Exact 46.47s compacted snapshot:", res_comp["snap_4647"])
    print("Compacted sink vs activity consistency:", res_comp["sink_activity_consistency"])
    print("Experimental comparison summary:", json.dumps(exp_metrics, indent=2))

    csv_out = OUT_DIR / "slab_gasification_timeseries.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s", "T_surf_undep_K", "T_back_undep_K", "J0_undep_g_m2_s", "Jchi_passive_g_m2_s",
            "E_source_pct", "E_bulk_pct", "cov_term_pct", "B_s_pct", "Jensen_upper_pct",
            "C90_pot_pct", "d90_pot_um", "max_cell_loss_pct",
            "Jchi_compacted_g_m2_s", "mass_frac_compacted", "T_back_compacted_K",
            "Jchi_fixed_g_m2_s", "mass_frac_fixed"
        ])
        for i in range(len(res_undep["t_s"])):
            w.writerow([
                f"{res_undep['t_s'][i]:.1f}",
                f"{res_undep['T_surf_K'][i]:.4f}",
                f"{res_undep['T_back_K'][i]:.4f}",
                f"{res_undep['J0_g_m2_s'][i]:.6f}",
                f"{res_undep['Jchi_g_m2_s'][i]:.6f}",
                f"{res_undep['E_source_pct'][i]:.6f}",
                f"{res_undep['E_bulk_pct'][i]:.6f}",
                f"{res_undep['cov_term_pct'][i]:.6f}",
                f"{res_undep['B_s_pct'][i]:.6f}",
                f"{res_undep['Jensen_pct'][i]:.6f}",
                f"{res_undep['C90_pot_pct'][i]:.6f}",
                f"{res_undep['d90_pot_um'][i]:.4f}",
                f"{res_undep['max_loss_pct'][i]:.6f}",
                f"{res_comp['Jchi_g_m2_s'][i]:.6f}",
                f"{res_comp['mass_frac'][i]:.6f}",
                f"{res_comp['T_back_K'][i]:.4f}",
                f"{res_fixed['Jchi_g_m2_s'][i]:.6f}",
                f"{res_fixed['mass_frac'][i]:.6f}",
            ])

    checkpoints = {}
    for t_c in (25.0, 30.0, 40.0, 41.3, 42.5, 46.4, 46.5, 51.6, 58.1, 60.0, 80.0, 106.4, 350.0, 367.9):
        idx = int(round(t_c / 0.1))
        checkpoints[f"{t_c:.1f}s"] = {
            "T_surf_K": round(float(res_undep["T_surf_K"][idx]), 3),
            "T_back_K": round(float(res_undep["T_back_K"][idx]), 3),
            "J0_g_m2_s": round(float(res_undep["J0_g_m2_s"][idx]), 4),
            "Jchi_passive_g_m2_s": round(float(res_undep["Jchi_g_m2_s"][idx]), 4),
            "Jchi_compacted_g_m2_s": round(float(res_comp["Jchi_g_m2_s"][idx]), 4),
            "Jchi_fixed_g_m2_s": round(float(res_fixed["Jchi_g_m2_s"][idx]), 4),
            "E_source_pct": round(float(res_undep["E_source_pct"][idx]), 4),
            "E_bulk_pct": round(float(res_undep["E_bulk_pct"][idx]), 4),
            "cov_term_pct": round(float(res_undep["cov_term_pct"][idx]), 4),
            "B_s_pct": round(float(res_undep["B_s_pct"][idx]), 4),
            "Jensen_pct": round(float(res_undep["Jensen_pct"][idx]), 4),
            "C90_pot_pct": round(float(res_undep["C90_pot_pct"][idx]), 4),
            "d90_pot_um": round(float(res_undep["d90_pot_um"][idx]), 2),
            "max_loss_pct": round(float(res_undep["max_loss_pct"][idx]), 4),
        }

    summary = {
        "benchmark": "1D MaCFP PMMA slab under 50 kW/m2 anaerobic radiant heating (NIST Gasification Apparatus)",
        "commit": "579b88265a329878a68e1161a8b9ceb5e62a9979",
        "parameters": {
            "source_file": "MaCFP_PMMA_NIST.json (2021 NIST TGA 1-step specification) + README.md Table 4 Kaowool PM backing",
            "A_per_s": 2.85e13,
            "E_J_per_mol": 191000.0,
            "reaction_order_n": 1.0,
            "char_yield_nu_s": 0.0,
            "volatile_yield_K_e": 1.0,
            "rho_0_kg_m3": 1100.0,
            "cp_pmma_J_kg_K": 2200.0,
            "k_pmma_W_m_K": 0.20,
            "delta_h_py_J_kg": 8.70e5,
            "surface_emissivity_absorptivity": 0.90,
            "radiative_absorption_mode": "surface-only (Absorption='inf' at cell i=0)",
            "q_steady_area_avg_W_m2": 49300.0,
            "q_star_t_knots_s": [0.0, 5.5, 35.0, 90.0, 150.0, 210.0, 270.0, 450.0, 1000.0],
            "q_star_values": [0.9300, 0.9401, 0.9598, 0.9831, 0.9946, 0.9987, 1.0000, 1.0004, 1.0004],
            "q_star_table1_interval_verification": q_star_verification,
            "h_conv_W_m2_K": 10.0,
            "T_init_K": 292.6,
            "T_wall_K": 289.15,
            "T_gas_K": 298.15,
            "kaowool_T_knots_K": [533.15, 811.15, 1089.15, 1366.15],
            "kaowool_k_knots_W_m_K": [0.0576, 0.0850, 0.1250, 0.1830],
            "kaowool_rho_kg_m3": 256.0,
            "kaowool_cp_J_kg_K": 1070.0,
            "L_pmma_rep_mm": 5.90,
            "L_ins_total_mm": 28.52,
            "L_ins_components_mm": "5.72 mm bonded high-temperature cement/Kaowool PM + 22.80 mm Kaowool PM stack",
            "backing_28p52_vs_22p80_max_diff": {
                "max_abs_dMLR_0_350s_g_m2_s": max_diff_mlr_backing,
                "max_abs_dT_back_0_350s_K": max_diff_tback_backing,
            },
            "N_pmma": 120,
            "first_cell_dy_um": 10.0,
            "r_p_5p90mm": round(res_comp["r_p"], 12),
            "N_ins": 60,
            "first_ins_dy_um": 80.0,
            "r_ins_28p52mm": round(res_comp["r_ins"], 12),
            "interface_T_back_initial_weights": {
                "w_p_0": round(res_comp["w_p_0"], 6),
                "w_ins_0": round(res_comp["w_ins_0"], 6),
                "max_abs_diff_res_vs_arith_0_210s_K": round(max_tback_res_vs_arith_210s, 4),
            },
            "dt_s": 0.01,
            "t_end_s": 450.0,
        },
        "exact_46p47s_undep": res_undep["snap_4647"],
        "exact_46p47s_compacted": res_comp["snap_4647"],
        "sink_activity_consistency": res_comp["sink_activity_consistency"],
        "crossings": res_undep["crossings"],
        "checkpoints": checkpoints,
        "mesh_dt_sensitivity": grid_study,
        "experimental_comparison": exp_metrics,
    }
    (OUT_DIR / "slab_gasification_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "slab_gasification_grid_curves.json").write_text(json.dumps(grid_curves), encoding="utf-8")
    print("Saved updated slab_gasification_metrics.json, slab_gasification_grid_curves.json, and slab_gasification_timeseries.csv")


if __name__ == "__main__":
    main()
