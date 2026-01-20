"""
  Esempio:
    G(s)=100/((s+10)(s+20))  -> num: 100   den: 1, 30, 200
    25/(s(1+s/10)(1+s/20))   -> prima semplifica:
       25 / ( s * (1+s/10) * (1+s/20) )
       = 25 / ( s * ((s+10)/10) * ((s+20)/20) )
       = 25*200 / ( s (s+10)(s+20) )
       = 5000 / ( s^3 + 30 s^2 + 200 s )
      -> num: 5000   den: 1, 30, 200, 0

- Inserisci anche: e(inf) (rampa unitaria), S% (overshoot), ts (tempo di salita)
- Premendo "Calcola" il programma:
    Step 1) Progetta R1 
    Step 2) ζ da S% e φm da ζ
    Step 3) ωn da ts, poi ωc
    Step 4) Ricava |R2(jωc)| e arg R2(jωc)
    Step 5-6) Sintetizza R2 come rete anticipatrice (lead) o attenuatrice (lag)
  e stampa TUTTI i passaggi nel riquadro testuale.
- Inoltre disegna i Bode di:
    * G(jω)
    * R1(jω)G(jω)
    * L(jω)=R(jω)G(jω)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from math import pi
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ----------------------------
# Utility: TF evaluation
# ----------------------------
def poly_eval_desc(coeffs, s):
    val = 0.0 + 0.0j
    for c in coeffs:
        val = val * s + c
    return val

def tf_eval(num, den, s):
    return poly_eval_desc(num, s) / poly_eval_desc(den, s)

def tf_mag_phase(num, den, w):
    s = 1j * w
    H = tf_eval(num, den, s)
    mag = abs(H)
    mag_db = 20.0 * np.log10(mag) if mag > 0 else -np.inf
    ph = np.angle(H, deg=True)
    return mag, mag_db, ph

def wrap_phase_deg(ph):
    return (ph + 180) % 360 - 180

def count_poles_at_origin(den, tol=1e-12):
    den = np.array(den, dtype=float)
    k = 0
    for c in den[::-1]:
        if abs(c) < tol:
            k += 1
        else:
            break
    return k

def dc_gain(num, den):
    if abs(den[-1]) < 1e-15:
        return np.inf
    return num[-1] / den[-1]


# ----------------------------
# Step response (scalino unitario) per G(s) e uscita del sistema
# ----------------------------
def tf_to_ss_controllable(num, den):
    """
    Trasforma una TF num/den (coeff. decrescenti) in forma di stato (canonica controllabile).
    Richiede TF propria (deg(num) <= deg(den)).
    """
    den = np.array(den, dtype=float)
    num = np.array(num, dtype=float)

    if abs(den[0]) < 1e-15:
        raise ValueError("Denominatore non valido (coeff. principale nullo).")

    # Normalizza denominatore a monico
    den = den / den[0]
    num = num / den[0]

    n = len(den) - 1  # ordine
    if n < 1:
        # sistema statico: y = (num0/den0) u
        D = float(num[-1] / den[-1]) if abs(den[-1]) > 1e-15 else np.inf
        A = np.zeros((0, 0))
        B = np.zeros((0, 1))
        C = np.zeros((1, 0))
        return A, B, C, np.array([[D]], dtype=float)

    # Rendi num della stessa lunghezza (n+1)
    if len(num) < n + 1:
        num = np.concatenate([np.zeros(n + 1 - len(num)), num])
    elif len(num) > n + 1:
        raise ValueError("TF impropria (grado numeratore > grado denominatore): non gestita.")

    # Matrice companion
    A = np.zeros((n, n), dtype=float)
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]

    B = np.zeros((n, 1), dtype=float)
    B[-1, 0] = 1.0

    # Forma canonica: C e D da divisione polinomiale (qui deg(num)<=deg(den))
    D = float(num[0])  # termine di grado n nel numeratore (dopo padding)
    # C: (b1 - a1*D, b2 - a2*D, ..., bn - an*D) con ordine coerente alla companion
    # dove num = [b0, b1, ..., bn], den = [1, a1, ..., an]
    a = den[1:]
    b = num[1:]
    C = (b - a * D).reshape(1, -1)[:, ::-1]  # ribalta per l'ordine della companion

    return A, B, C, np.array([[D]], dtype=float)

def step_response_rk4(num, den, t_end=None, n_points=2000):
    """
    Risposta al gradino unitario usando integrazione RK4 su modello in spazio di stato.
    """
    A, B, C, D = tf_to_ss_controllable(num, den)

    # Caso statico
    if A.size == 0:
        if t_end is None:
            t_end = 5.0
        t = np.linspace(0.0, float(t_end), int(n_points))
        y = np.ones_like(t) * float(D[0, 0])
        return t, y

    # Scelta automatica di t_end in base ai poli (se possibile)
    if t_end is None:
        try:
            poles = np.roots(np.array(den, dtype=float))
            stable_reals = [abs(p.real) for p in poles if p.real < -1e-9]
            if stable_reals:
                tau_dom = 1.0 / min(stable_reals)
                t_end = 8.0 * tau_dom
            else:
                t_end = 5.0
        except Exception:
            t_end = 5.0

    t_end = float(max(t_end, 1e-6))
    t = np.linspace(0.0, t_end, int(n_points))
    dt = t[1] - t[0]

    x = np.zeros((A.shape[0],), dtype=float)
    y = np.zeros_like(t, dtype=float)

    u = 1.0  # scalino unitario

    def f(xv):
        return (A @ xv) + (B[:, 0] * u)

    for k in range(len(t)):
        y[k] = float(C @ x + D[0, 0] * u)

        if k == len(t) - 1:
            break

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y


# ----------------------------
# Global links 
# ----------------------------
def zeta_from_overshoot(S_pct):
    S = S_pct / 100.0
    if not (0 < S < 1):
        raise ValueError("S% deve essere tra 0 e 100 (esclusi).")
    lnS = np.log(S)
    z = -lnS / np.sqrt(pi**2 + lnS**2)
    return float(z)

def phase_margin_from_zeta(z):
    if z <= 0 or z >= 1:
        raise ValueError("Relazione valida per 0<ζ<1.")
    term = np.sqrt(-2*z*z + np.sqrt(4*z**4 + 1.0))
    A = term / (2*z)
    phi_m = 90.0 - (180.0/pi)*np.arctan(A)
    return float(phi_m)

def wn_from_rise_time(ts, z):
    if ts <= 0:
        raise ValueError("ts deve essere >0.")
    if not (0 < z < 1):
        raise ValueError("Formula pensata per 0<ζ<1.")
    a = np.sqrt(1 - z*z)
    value = (1.0/a) * (pi - np.arctan(a/z))
    wn = value / ts
    return float(wn)

def wc_from_wn_zeta(wn, z):
    term = np.sqrt(-2*z*z + np.sqrt(4*z**4 + 1.0))
    wc = wn * term
    return float(wc)


# ----------------------------
# Step 1: Design R1 for ramp error
# ----------------------------
def design_R1_for_ramp(G_num, G_den, e_inf):
    if e_inf <= 0:
        raise ValueError("e(inf) deve essere >0.")
    plant_type = count_poles_at_origin(G_den)
    needed_integrators = max(0, 1 - plant_type)

    if needed_integrators == 0:
        # Plant already type>=1, choose R1=K (gain) to meet ramp error via Kv
        eps = 1e-6
        Kv_base = (eps * tf_eval(G_num, G_den, eps)).real
        if Kv_base <= 0:
            raise ValueError("Non riesco a stimare Kv_base. Controlla G(s).")
        K = 1.0 / (e_inf * Kv_base)
        R1_num = [K]
        R1_den = [1.0]
        return R1_num, R1_den, plant_type, needed_integrators, K, Kv_base

    if needed_integrators == 1:
        G0 = dc_gain(G_num, G_den)
        if not np.isfinite(G0) or abs(G0) < 1e-15:
            raise ValueError("G(0) non finito o nullo: caso non gestito con R1=K/s.")
        K = 1.0 / (e_inf * G0)
        R1_num = [K]
        R1_den = [1.0, 0.0]  # s
        return R1_num, R1_den, plant_type, needed_integrators, K, G0

    raise ValueError("Caso con più integratori mancanti non gestito in questa versione.")


# ----------------------------
# R2 synthesis at wc
# ----------------------------
def lead_mag_at_x_m(x, m):
    return float(np.sqrt(1+x*x) / np.sqrt(1+(x/m)**2))

def solve_x_for_lead_phase(phi_deg, m):
    # tan(phi) = (x(1-1/m)) / (1 + x^2/m)
    t = np.tan(phi_deg * pi/180.0)
    a = (t / m)
    b = -(1.0 - 1.0/m)
    c = t
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    if abs(a) < 1e-15:
        return None
    x1 = (-b + np.sqrt(disc)) / (2*a)
    x2 = (-b - np.sqrt(disc)) / (2*a)
    candidates = [x for x in (x1, x2) if x > 0]
    if not candidates:
        return None
    return float(min(candidates))

def design_R2(G_num, G_den, R1_num, R1_den, wc, phi_m_deg):
    # L1 = R1*G at wc
    R1G_num = np.polymul(R1_num, G_num)
    R1G_den = np.polymul(R1_den, G_den)
    _, mag_R1G_db, ph_R1G = tf_mag_phase(R1G_num, R1G_den, wc)
    ph_R1G = wrap_phase_deg(ph_R1G)

    # Required R2 magnitude at wc
    R2_mag_db_req = -mag_R1G_db
    R2_mag_req = 10**(R2_mag_db_req/20.0)

    # Required R2 phase at wc 
    R2_phase_req = -ph_R1G - 180.0 + phi_m_deg
    R2_phase_req = wrap_phase_deg(R2_phase_req)

    use_lead = (R2_phase_req > 0.0)

    if use_lead:
        # choose a "nice" m, otherwise continuous
        m_candidates = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
        chosen = None
        for m in m_candidates:
            phi_max = np.degrees(np.arcsin((m-1)/(m+1)))
            if phi_max >= (R2_phase_req + 1.0):
                chosen = m
                break
        if chosen is None:
            s = np.sin(np.radians(R2_phase_req + 2.0))
            chosen = (1+s)/(1-s)
        m = float(chosen)

        x = solve_x_for_lead_phase(R2_phase_req, m)
        if x is None:
            x = float(np.sqrt(m))  # fallback: peak near wc
        tau = x / wc

        mag0 = lead_mag_at_x_m(x, m)
        K2 = R2_mag_req / mag0

        # R2(s) = K2*(1+tau s)/(1+(tau/m)s)
        R2_num = [K2*tau, K2]
        R2_den = [tau/m, 1.0]

        info = {"type": "ANTICIPATRICE", "m": m, "x": x, "tau": tau, "mag0": mag0, "K2": K2}
        return R2_num, R2_den, mag_R1G_db, ph_R1G, R2_mag_db_req, R2_phase_req, info

    else:
        # lag network for attenuation
        m = float(1.0 / R2_mag_req) if R2_mag_req > 0 else 10.0
        if m < 1.01:
            m = 1.01

        # choose big x to keep phase small
        x = float(max(10*m, 50.0))
        tau = x / wc

        mag0 = float(np.sqrt(1+(x/m)**2) / np.sqrt(1+x**2))
        phi0 = float((180.0/pi)*(np.arctan(x/m) - np.arctan(x)))

        K2 = R2_mag_req / mag0

        # R2(s)=K2*(1+(tau/m)s)/(1+tau s)
        R2_num = [K2*(tau/m), K2]
        R2_den = [tau, 1.0]

        info = {"type": "ATTENUATRICE", "m": m, "x": x, "tau": tau, "mag0": mag0, "phi0": phi0, "K2": K2}
        return R2_num, R2_den, mag_R1G_db, ph_R1G, R2_mag_db_req, R2_phase_req, info


# ----------------------------
# Text helpers
# ----------------------------
def fmt_poly(coeffs):
    coeffs = np.array(coeffs, dtype=float)
    # Short format list
    return "[" + ", ".join(f"{c:.6g}" for c in coeffs) + "]"

def parse_coeffs(text):
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("Lista coefficienti vuota.")
    return [float(p) for p in parts]


# ----------------------------
# GUI app
# ----------------------------
class ControlDesignerApp:
    def __init__(self, root):
        self.root = root
        root.title("Sintesi R1-R2 Sergio Ginex")

        root.geometry("1280x720")     # 16:9
        root.minsize(960, 540)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)  # area centrale (paned)

        # Inputs (in alto)
        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="G(s) Numeratore (coeff. decrescenti):").grid(row=0, column=0, sticky="w")
        self.num_entry = ttk.Entry(frm, width=60)
        self.num_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.num_entry.insert(0, "100")

        ttk.Label(frm, text="G(s) Denominatore (coeff. decrescenti):").grid(row=1, column=0, sticky="w")
        self.den_entry = ttk.Entry(frm, width=60)
        self.den_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.den_entry.insert(0, "1, 30, 200")

        ttk.Label(frm, text="e(∞) (rampa unitaria):").grid(row=2, column=0, sticky="w")
        self.einf_entry = ttk.Entry(frm, width=20)
        self.einf_entry.grid(row=2, column=1, sticky="w", padx=5)
        self.einf_entry.insert(0, "0.04")

        ttk.Label(frm, text="S% (overshoot):").grid(row=3, column=0, sticky="w")
        self.S_entry = ttk.Entry(frm, width=20)
        self.S_entry.grid(row=3, column=1, sticky="w", padx=5)
        self.S_entry.insert(0, "55")

        ttk.Label(frm, text="t_s (s):").grid(row=4, column=0, sticky="w")
        self.ts_entry = ttk.Entry(frm, width=20)
        self.ts_entry.grid(row=4, column=1, sticky="w", padx=5)
        self.ts_entry.insert(0, "0.123")

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Button(btns, text="Calcola", command=self.compute).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Reset", command=self.reset_example).grid(row=0, column=1, padx=5)

        # Paned window (sinistra testo, destra grafici)
        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Pannello sinistro: testo con scrollbar
        out_frame = ttk.Frame(paned, padding=10)
        paned.add(out_frame, weight=1)

        ttk.Label(out_frame, text="Procedimento:").grid(row=0, column=0, sticky="w")
        self.text = tk.Text(out_frame, height=18, wrap="word")
        yscroll = ttk.Scrollbar(out_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        self.text.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")
        out_frame.rowconfigure(1, weight=1)
        out_frame.columnconfigure(0, weight=1)

        # Pannello destro: notebook con Bode e Step
        plot_frame = ttk.Frame(paned, padding=10)
        paned.add(plot_frame, weight=2)
        plot_frame.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        nb = ttk.Notebook(plot_frame)
        nb.grid(row=1, column=0, sticky="nsew")

        bode_tab = ttk.Frame(nb)
        step_tab = ttk.Frame(nb)
        nb.add(bode_tab, text="Bode")
        nb.add(step_tab, text="Scalino")

        # Tab Bode
        bode_tab.rowconfigure(1, weight=1)
        bode_tab.columnconfigure(0, weight=1)
        ttk.Label(bode_tab, text="Grafici di Bode (G, R1G, L=RG):").grid(row=0, column=0, sticky="w")

        self.fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.ax_mag = self.fig.add_subplot(211)
        self.ax_ph = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=bode_tab)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Tab Step
        step_tab.rowconfigure(1, weight=1)
        step_tab.columnconfigure(0, weight=1)
        ttk.Label(step_tab, text="Risposta allo scalino unitario (G e L):").grid(row=0, column=0, sticky="w")

        self.fig_step = plt.Figure(figsize=(9, 4.5), dpi=100)
        self.ax_step_G = self.fig_step.add_subplot(211)
        self.ax_step_L = self.fig_step.add_subplot(212)

        self.canvas_step = FigureCanvasTkAgg(self.fig_step, master=step_tab)
        self.canvas_step.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self._last_results = None

    def reset_example(self):
        self.num_entry.delete(0, tk.END)
        self.num_entry.insert(0, "100")
        self.den_entry.delete(0, tk.END)
        self.den_entry.insert(0, "1, 30, 200")
        self.einf_entry.delete(0, tk.END)
        self.einf_entry.insert(0, "0.04")
        self.S_entry.delete(0, tk.END)
        self.S_entry.insert(0, "55")
        self.ts_entry.delete(0, tk.END)
        self.ts_entry.insert(0, "0.123")
        self.text.delete("1.0", tk.END)
        self.clear_plots()

    def clear_plots(self):
        self.ax_mag.clear()
        self.ax_ph.clear()
        self.ax_mag.set_ylabel("Modulo (dB)")
        self.ax_ph.set_ylabel("Fase (deg)")
        self.ax_ph.set_xlabel("ω (rad/s)")
        self.canvas.draw()

        self.ax_step_G.clear()
        self.ax_step_L.clear()
        self.ax_step_G.set_ylabel("y_G(t)")
        self.ax_step_L.set_ylabel("y_L(t)")
        self.ax_step_L.set_xlabel("t (s)")
        self.canvas_step.draw()

    def log(self, s):
        self.text.insert(tk.END, s + "\n")
        self.text.see(tk.END)

    def compute(self):
        try:
            G_num = parse_coeffs(self.num_entry.get())
            G_den = parse_coeffs(self.den_entry.get())
            e_inf = float(self.einf_entry.get())
            S_pct = float(self.S_entry.get())
            ts = float(self.ts_entry.get())

            # Basic checks
            if len(G_den) < 2:
                raise ValueError("Denominatore troppo corto.")
            if abs(G_den[0]) < 1e-15:
                raise ValueError("Il coefficiente principale del denominatore non può essere zero.")

            self.text.delete("1.0", tk.END)

            self.log("="*90)
            self.log("SINTESI NEL DOMINIO DI ω")
            self.log("="*90)
            self.log(f"G(s): num={fmt_poly(G_num)}  den={fmt_poly(G_den)}")
            self.log(f"Specifiche: e(∞)≤{e_inf},  S%≤{S_pct}%,  t_s={ts} s")
            self.log("")

            # STEP 1
            self.log("-"*90)
            self.log("STEP 1) Specifiche statiche: progetto di R1")
            self.log("-"*90)

            R1_num, R1_den, plant_type, needed_int, K, G0_or_Kv = design_R1_for_ramp(G_num, G_den, e_inf)
            self.log(f"Tipo impianto (poli in origine) = {plant_type}")
            self.log(f"Integratori da aggiungere per tipo>=1 = {needed_int}")
            if needed_int == 1:
                self.log("Scelta: R1(s)=K/s")
                self.log(f"G(0) = {G0_or_Kv:.6g}")
                self.log(f"K = 1/(e(∞)*G(0)) = 1/({e_inf}*{G0_or_Kv:.6g}) = {K:.6g}")
            else:
                self.log("Impianto già di tipo>=1: scelgo R1(s)=K (solo guadagno)")
                self.log(f"Stima base per Kv: {G0_or_Kv:.6g}")
                self.log(f"K = 1/(e(∞)*Kv_base) = {K:.6g}")
            self.log(f"R1(s): num={fmt_poly(R1_num)}  den={fmt_poly(R1_den)}")
            self.log("")

            # STEP 2
            self.log("-"*90)
            self.log("STEP 2) Specifiche dinamiche: ζ da S% e φm da ζ (legami globali)")
            self.log("-"*90)
            z = zeta_from_overshoot(S_pct)
            phi_m = phase_margin_from_zeta(z)
            self.log(f"ζ = {z:.6f}")
            self.log(f"φm = {phi_m:.6f}°")
            self.log("")

            # STEP 3
            self.log("-"*90)
            self.log("STEP 3) Tempo di salita: ωn da t_s e ζ; poi ωc da ωn e ζ")
            self.log("-"*90)
            wn = wn_from_rise_time(ts, z)
            wc = wc_from_wn_zeta(wn, z)
            self.log(f"ωn = {wn:.6f} rad/s")
            self.log(f"ωc = {wc:.6f} rad/s")
            self.log("")

            # STEP 4-6
            self.log("-"*90)
            self.log("STEP 4) Vincoli su R2(jωc): modulo e fase per attraversamento e margine di fase")
            self.log("-"*90)

            R2_num, R2_den, mag_R1G_db, ph_R1G, R2_mag_db_req, R2_phase_req, info = design_R2(
                G_num, G_den, R1_num, R1_den, wc, phi_m
            )

            self.log(f"A ωc={wc:.6f} rad/s:")
            self.log(f"|R1(jωc)G(jωc)|_dB = {mag_R1G_db:.6f} dB")
            self.log(f"arg(R1(jωc)G(jωc)) = {ph_R1G:.6f}°")
            self.log("")
            self.log("Formule:")
            self.log(f"|R2(jωc)|_dB = -|R1(jωc)G(jωc)|_dB = {R2_mag_db_req:.6f} dB")
            self.log(f"arg R2(jωc)  = -arg(R1G)-180°+φm = {R2_phase_req:.6f}°")
            self.log("")

            self.log("-"*90)
            self.log("STEP 5-6) Sintesi di R2(s)")
            self.log("-"*90)
            self.log(f"Tipo rete scelta: {info['type'].upper()}")
            self.log(f"m = {info['m']:.6g}")
            self.log(f"ωc·τ = x = {info['x']:.6g}")
            self.log(f"τ = x/ωc = {info['tau']:.6g} s")
            if info["type"] == "ANTICIPATRICE":
                self.log(f"|R2_0(jωc)| (senza K2) = {info['mag0']:.6g}  ({20*np.log10(info['mag0']):.4f} dB)")
                self.log(f"K2 (guadagno addizionale) = {info['K2']:.6g}  ({20*np.log10(info['K2']):.4f} dB)")
            else:
                self.log(f"|R2_0(jωc)| (senza K2) = {info['mag0']:.6g}  ({20*np.log10(info['mag0']):.4f} dB)")
                self.log(f"arg(R2_0(jωc)) (senza K2) = {info['phi0']:.4f}°")
                self.log(f"K2 (guadagno addizionale) = {info['K2']:.6g}  ({20*np.log10(info['K2']):.4f} dB)")

            self.log(f"R2(s): num={fmt_poly(R2_num)}  den={fmt_poly(R2_den)}")
            self.log("")

            # Final controller
            R_num = np.polymul(R1_num, R2_num)
            R_den = np.polymul(R1_den, R2_den)
            self.log("-"*90)
            self.log("RISULTATO: R(s)=R1(s)R2(s)")
            self.log("-"*90)
            self.log(f"R(s): num={fmt_poly(R_num)}  den={fmt_poly(R_den)}")
            self.log("")

            # Verify at wc
            L_num = np.polymul(R_num, G_num)
            L_den = np.polymul(R_den, G_den)
            magL, magL_db, phL = tf_mag_phase(L_num, L_den, wc)
            PM_est = 180.0 + wrap_phase_deg(phL)

            self.log("Verifica rapida a ωc:")
            self.log(f"|L(jωc)| = {magL:.6g}  ({magL_db:.4f} dB)")
            self.log(f"arg L(jωc) = {wrap_phase_deg(phL):.4f}°  -> PM ≈ {PM_est:.4f}° (target {phi_m:.4f}°)")
            self.log("")

            # >>> MODIFICA RICHIESTA: y_L(t) deve essere l'uscita del sistema retroazionato
            # T(s) = L(s) / (1 + L(s)) con retroazione unitaria
            T_num = L_num
            T_den = np.polyadd(L_den, L_num)
            # <<< FINE MODIFICA RICHIESTA <<<

            # Save results for plotting
            self._last_results = {
                "G": (G_num, G_den),
                "R1": (R1_num, R1_den),
                "R2": (R2_num, R2_den),
                "R": (R_num, R_den),
                "L": (L_num, L_den),
                "T": (T_num, T_den),   # >>> MODIFICA RICHIESTA <<<
                "wc": wc
            }
            self.plot_bode()
            self.plot_step()

        except Exception as ex:
            messagebox.showerror("Errore", str(ex))

    def plot_bode(self):
        if not self._last_results:
            return
        self.ax_mag.clear()
        self.ax_ph.clear()

        G_num, G_den = self._last_results["G"]
        R1_num, R1_den = self._last_results["R1"]
        R_num, R_den = self._last_results["R"]
        wc = self._last_results["wc"]

        # Frequency grid around wc
        wmin = max(1e-2, wc/100)
        wmax = wc*100
        w = np.logspace(np.log10(wmin), np.log10(wmax), 600)

        def bode_arrays(num, den):
            H = []
            for wi in w:
                val = tf_eval(num, den, 1j*wi)
                H.append(val)
            H = np.array(H)

            mags_db = 20 * np.log10(np.abs(H))

            ph_rad = np.unwrap(np.angle(H))      # fase continua (radianti)
            ph_deg = np.degrees(ph_rad)          # in gradi

            return mags_db, ph_deg

        # G
        G_mag, G_ph = bode_arrays(G_num, G_den)

        # R1G
        R1G_num = np.polymul(R1_num, G_num)
        R1G_den = np.polymul(R1_den, G_den)
        R1G_mag, R1G_ph = bode_arrays(R1G_num, R1G_den)

        # L = R*G
        L_num = np.polymul(R_num, G_num)
        L_den = np.polymul(R_den, G_den)
        L_mag, L_ph = bode_arrays(L_num, L_den)

        # Plot
        self.ax_mag.semilogx(w, G_mag, label="|G(jω)|")
        self.ax_mag.semilogx(w, R1G_mag, label="|R1(jω)G(jω)|")
        self.ax_mag.semilogx(w, L_mag, label="|L(jω)=R(jω)G(jω)|")
        self.ax_mag.axvline(wc, linestyle="--", linewidth=1.0)
        self.ax_mag.set_ylabel("Modulo (dB)")
        self.ax_mag.grid(True, which="both", ls=":")
        self.ax_mag.legend(loc="best")

        self.ax_ph.semilogx(w, G_ph, label="∠G(jω)")
        self.ax_ph.semilogx(w, R1G_ph, label="∠R1G(jω)")
        self.ax_ph.semilogx(w, L_ph, label="∠L(jω)")
        self.ax_ph.axvline(wc, linestyle="--", linewidth=1.0)
        self.ax_ph.set_ylabel("Fase (deg)")
        self.ax_ph.set_xlabel("ω (rad/s)")
        self.ax_ph.grid(True, which="both", ls=":")
        self.ax_ph.legend(loc="best")

        self.fig.tight_layout()
        self.canvas.draw()

    # ----------------------------
    # Risposta allo scalino unitario di G(s) e dell'uscita totale retroazionata
    # ----------------------------
    def plot_step(self):
        if not self._last_results:
            return

        self.ax_step_G.clear()
        self.ax_step_L.clear()

        G_num, G_den = self._last_results["G"]

        # >>> MODIFICA RICHIESTA: usa T(s) (anello chiuso) al posto di L(s)
        T_num, T_den = self._last_results["T"]
        # <<< FINE MODIFICA RICHIESTA <<<

        # Risposte al gradino
        tG, yG = step_response_rk4(G_num, G_den, t_end=None, n_points=2000)
        tL, yL = step_response_rk4(T_num, T_den, t_end=None, n_points=2000)

        self.ax_step_G.plot(tG, yG, label="y_G(t) (step unitario)")
        self.ax_step_G.set_ylabel("y_G(t)")
        self.ax_step_G.grid(True, ls=":")
        self.ax_step_G.legend(loc="best")

        self.ax_step_L.plot(tL, yL, label="y_L(t) (step unitario)")
        self.ax_step_L.set_ylabel("y_L(t)")
        self.ax_step_L.set_xlabel("t (s)")
        self.ax_step_L.grid(True, ls=":")
        self.ax_step_L.legend(loc="best")

        self.fig_step.tight_layout()
        self.canvas_step.draw()


def main():
    root = tk.Tk()
    app = ControlDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
